//! Qwen3 throughput benchmarks for `renderers-core`.
//!
//! Needs a real tokenizer.json on disk because the benchmarks measure
//! end-to-end render/parse latency (the tokenizer is on the hot path).
//! Set `BENCH_TOKENIZER=/path/to/tokenizer.json` before running:
//!
//! ```bash
//! BENCH_TOKENIZER=/path/to/qwen3-8b/tokenizer.json \
//!     cargo bench -p renderers-cli
//! ```
//!
//! When `BENCH_TOKENIZER` is unset the benches return early without
//! failing — they're informational, not a CI gate.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use renderers_core::Renderer;
use renderers_core::families::Qwen3Renderer;
use renderers_core::tokenizer::Tokenizer;
use renderers_core::types::{Content, Message};

fn tokenizer() -> Option<Tokenizer> {
    let path = std::env::var("BENCH_TOKENIZER").ok()?;
    match Tokenizer::from_file(&path) {
        Ok(t) => Some(t),
        Err(e) => {
            eprintln!("bench skipped — couldn't load tokenizer at {path}: {e}");
            None
        }
    }
}

fn text_msg(role: &str, content: &str) -> Message {
    Message {
        role: role.to_string(),
        content: Content::Text(content.to_string()),
        ..Default::default()
    }
}

fn typical_conversation() -> Vec<Message> {
    vec![
        text_msg(
            "system",
            "You are a helpful assistant that calls tools when needed.",
        ),
        text_msg(
            "user",
            "Plan a weekend trip to Lisbon for two; we like food and walking.",
        ),
        text_msg(
            "assistant",
            "I'll help. First, let me check the weather and find some restaurants.",
        ),
        text_msg("user", "Sounds good — go ahead."),
        text_msg(
            "assistant",
            "Here's a plan: Friday evening tapas at Time Out Market, \
             Saturday morning walk through Alfama, Saturday lunch at \
             Ramiro (seafood), Saturday afternoon Belém pastéis, \
             Sunday morning São Jorge castle, Sunday lunch at Cervejaria \
             Trindade.",
        ),
    ]
}

fn bench_render_ids(c: &mut Criterion) {
    let Some(tok) = tokenizer() else {
        return;
    };
    let renderer = Qwen3Renderer::new(tok).expect("build Qwen3 renderer");
    let messages = typical_conversation();
    let mut group = c.benchmark_group("qwen3");
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("render_ids/5_turn_text", |b| {
        b.iter(|| {
            let ids = renderer
                .render_ids(black_box(&messages), None, true)
                .expect("render_ids");
            black_box(ids);
        });
    });
    group.finish();
}

fn bench_parse_response(c: &mut Criterion) {
    let Some(tok) = tokenizer() else {
        return;
    };
    let renderer = Qwen3Renderer::new(tok).expect("build Qwen3 renderer");
    let messages = typical_conversation();
    // Render once to get a realistic completion-ish prefix; treat it
    // as a "completion" for the parse benchmark.
    let output = renderer.render(&messages, None, true).expect("render");
    let ids = output.token_ids;

    let mut group = c.benchmark_group("qwen3");
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("parse_response/no_tool_calls", |b| {
        b.iter(|| {
            let parsed = renderer.parse_response(black_box(&ids));
            black_box(parsed);
        });
    });
    group.finish();
}

fn bench_bridge(c: &mut Criterion) {
    let Some(tok) = tokenizer() else {
        return;
    };
    let renderer = Qwen3Renderer::new(tok).expect("build Qwen3 renderer");
    let messages = typical_conversation();
    let output = renderer.render(&messages, None, true).expect("render");
    let prev_prompt_ids = output.token_ids.clone();
    let prev_completion_ids: Vec<u32> = vec![];
    let new_messages = vec![text_msg(
        "user",
        "Add a kid-friendly option for Sunday morning.",
    )];

    let mut group = c.benchmark_group("qwen3");
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("bridge_to_next_turn/short_user_turn", |b| {
        b.iter(|| {
            let bridged = renderer
                .bridge_to_next_turn(
                    black_box(&prev_prompt_ids),
                    black_box(&prev_completion_ids),
                    black_box(&new_messages),
                    None,
                )
                .expect("bridge");
            black_box(bridged);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_render_ids,
    bench_parse_response,
    bench_bridge
);
criterion_main!(benches);
