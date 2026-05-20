//! `renderers-cli` — small dev tool that drives `renderers-core`
//! without going through Python.
//!
//! Designed for two use cases:
//!
//! 1. **Golden parity checking**: render a fixture JSON of messages
//!    against a tokenizer.json, emit the result as JSON, and `diff`
//!    against the Python reference output. The exit code is non-zero
//!    if the run fails — the comparison is left to the caller (the
//!    pytest harness does the actual diffing).
//! 2. **Manual prototyping**: try out new families / config changes
//!    without spinning up the `PyO3` wheel.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use renderers_core::Renderer;
use renderers_core::families::Qwen3Renderer;
use renderers_core::tokenizer::Tokenizer;
use renderers_core::types::{Message, ParsedToolCall, RenderedTokens, ToolArguments, ToolSpec};
use serde::Serialize;

/// Render and parse messages via `renderers-core`. Output is line-by-line
/// JSON on stdout for easy diffing.
#[derive(Debug, Parser)]
#[command(name = "renderers-cli", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Render a conversation to token ids + per-token message indices.
    Render(RenderArgs),

    /// Parse a completion's token ids into a structured response.
    Parse(ParseArgs),
}

/// Renderer families wired through to `renderers-core`. New families
/// land here as they're ported.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Family {
    Qwen3,
}

#[derive(Debug, Parser)]
struct RenderArgs {
    /// Renderer family to instantiate.
    #[arg(long, value_enum, default_value_t = Family::Qwen3)]
    family: Family,

    /// Path to a `tokenizer.json` file.
    #[arg(long)]
    tokenizer: PathBuf,

    /// Path to a JSON file containing a list of messages.
    #[arg(long)]
    messages: PathBuf,

    /// Path to a JSON file containing a list of tool specs.
    #[arg(long)]
    tools: Option<PathBuf>,

    /// Emit a trailing generation prompt (`<|im_start|>assistant\n` for
    /// Qwen3).
    #[arg(long)]
    gen_prompt: bool,
}

#[derive(Debug, Parser)]
struct ParseArgs {
    /// Renderer family to instantiate.
    #[arg(long, value_enum, default_value_t = Family::Qwen3)]
    family: Family,

    /// Path to a `tokenizer.json` file.
    #[arg(long)]
    tokenizer: PathBuf,

    /// JSON-encoded list of integer token ids
    /// (e.g. `'[151644, 8948, 198, ...]'`).
    #[arg(long)]
    token_ids: String,
}

#[derive(Serialize)]
struct RenderedJson {
    token_ids: Vec<u32>,
    message_indices: Vec<i32>,
}

impl From<RenderedTokens> for RenderedJson {
    fn from(r: RenderedTokens) -> Self {
        Self {
            token_ids: r.token_ids,
            message_indices: r.message_indices,
        }
    }
}

#[derive(Serialize)]
struct ParsedToolCallJson<'a> {
    raw: &'a str,
    name: Option<&'a str>,
    arguments: serde_json::Value,
    status: &'static str,
    token_span: Option<(usize, usize)>,
    id: Option<&'a str>,
}

impl<'a> From<&'a ParsedToolCall> for ParsedToolCallJson<'a> {
    fn from(p: &'a ParsedToolCall) -> Self {
        let args = match &p.arguments {
            None => serde_json::Value::Null,
            Some(ToolArguments::Object(v)) => v.clone(),
            Some(ToolArguments::Raw(s)) => serde_json::Value::String(s.clone()),
        };
        Self {
            raw: &p.raw,
            name: p.name.as_deref(),
            arguments: args,
            status: p.status.as_wire(),
            token_span: p.token_span.as_ref().map(|r| (r.start, r.end)),
            id: p.id.as_deref(),
        }
    }
}

#[derive(Serialize)]
struct ParsedJson<'a> {
    content: &'a str,
    reasoning_content: Option<&'a str>,
    tool_calls: Vec<ParsedToolCallJson<'a>>,
}

fn build_renderer(family: Family, tokenizer: Tokenizer) -> Result<Box<dyn Renderer>, String> {
    match family {
        Family::Qwen3 => Qwen3Renderer::new(tokenizer)
            .map(|r| Box::new(r) as Box<dyn Renderer>)
            .map_err(|e| e.to_string()),
    }
}

fn load_messages(path: &PathBuf) -> Result<Vec<Message>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("messages JSON: {e}"))
}

fn load_tools(path: &PathBuf) -> Result<Vec<ToolSpec>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("tools JSON: {e}"))
}

fn parse_token_ids(s: &str) -> Result<Vec<u32>, String> {
    let v: Vec<i64> = serde_json::from_str(s).map_err(|e| format!("token-ids JSON: {e}"))?;
    v.into_iter()
        .map(|t| u32::try_from(t).map_err(|_| format!("token id out of range: {t}")))
        .collect()
}

fn run_render(args: &RenderArgs) -> Result<(), String> {
    let tok = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| format!("load tokenizer {}: {e}", args.tokenizer.display()))?;
    let renderer = build_renderer(args.family, tok)?;
    let messages = load_messages(&args.messages)?;
    let tools = match args.tools.as_ref() {
        Some(p) => Some(load_tools(p)?),
        None => None,
    };
    let output = renderer
        .render(&messages, tools.as_deref(), args.gen_prompt)
        .map_err(|e| e.to_string())?;
    let json: RenderedJson = output.into();
    println!("{}", serde_json::to_string(&json).unwrap());
    Ok(())
}

fn run_parse(args: &ParseArgs) -> Result<(), String> {
    let tok = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| format!("load tokenizer {}: {e}", args.tokenizer.display()))?;
    let renderer = build_renderer(args.family, tok)?;
    let ids = parse_token_ids(&args.token_ids)?;
    let parsed = renderer.parse_response(&ids);
    let tool_calls: Vec<ParsedToolCallJson<'_>> = parsed
        .tool_calls
        .iter()
        .map(ParsedToolCallJson::from)
        .collect();
    let json = ParsedJson {
        content: &parsed.content,
        reasoning_content: parsed.reasoning_content.as_deref(),
        tool_calls,
    };
    println!("{}", serde_json::to_string(&json).unwrap());
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Render(args) => run_render(&args),
        Command::Parse(args) => run_parse(&args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}
