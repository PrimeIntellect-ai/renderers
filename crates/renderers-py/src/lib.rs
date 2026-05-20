//! Python bindings for `renderers-core`.
//!
//! The boundary is intentionally thin: one polymorphic `Renderer`
//! pyclass holds an `Arc<dyn renderers_core::Renderer>`; small result
//! pyclasses wrap `RenderedTokens` / `ParsedResponse` / `ParsedToolCall`
//! with `#[getter]` accessors. Argument unpacking is done by
//! `pythonize` so callers can pass plain dicts / lists for messages and
//! tools without per-field PyO3 conversion.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyType};

use renderers_core::families::{
    DeepSeekV3RendererBuilder, GlmRendererBuilder, Nemotron3RendererBuilder,
    Qwen35RendererBuilder, Qwen36RendererBuilder, Qwen3RendererBuilder,
};
use renderers_core::tokenizer::Tokenizer;
use renderers_core::types::{
    Message, ParsedResponse, ParsedToolCall, RenderedTokens, ToolArguments, ToolCallParseStatus,
    ToolSpec,
};
use renderers_core::Renderer as CoreRenderer;

fn render_err(e: renderers_core::types::RenderError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn invalid(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Decode a Python `list[dict]` of messages via pythonize.
fn parse_messages(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Message>> {
    let value: serde_json::Value = pythonize::depythonize(obj).map_err(|e| {
        invalid(format!("messages must be a list of dicts (decode failed: {e})"))
    })?;
    serde_json::from_value(value).map_err(|e| invalid(format!("messages shape mismatch: {e}")))
}

fn parse_tools(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<ToolSpec>>> {
    let Some(obj) = obj else { return Ok(None) };
    if obj.is_none() {
        return Ok(None);
    }
    let value: serde_json::Value = pythonize::depythonize(obj)
        .map_err(|e| invalid(format!("tools must be a list of dicts (decode failed: {e})")))?;
    let parsed: Vec<ToolSpec> = serde_json::from_value(value)
        .map_err(|e| invalid(format!("tools shape mismatch: {e}")))?;
    Ok(Some(parsed))
}

fn parse_u32_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    // Accept either a Python list of ints or a numpy-style sequence.
    let list = obj.downcast::<PyList>().map_err(|_| invalid("expected list[int]"))?;
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let v: i64 = item.extract()?;
        if v < 0 || v > u32::MAX as i64 {
            return Err(invalid(format!("token id out of range: {v}")));
        }
        out.push(v as u32);
    }
    Ok(out)
}

#[pyclass(name = "RenderedTokens", module = "renderers_native")]
#[derive(Clone)]
struct PyRenderedTokens {
    inner: RenderedTokens,
}

#[pymethods]
impl PyRenderedTokens {
    #[getter]
    fn token_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        // Cast u32 -> i64 for Python `int` compatibility. PyList::new is
        // the fastest path; per-element extract is unavoidable until
        // numpy support is added.
        PyList::new_bound(py, self.inner.token_ids.iter().map(|&t| t as i64))
    }

    #[getter]
    fn message_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::new_bound(py, self.inner.message_indices.iter().copied())
    }

    #[getter]
    fn multi_modal_data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner.multi_modal_data {
            Some(mm) => pythonize::pythonize(py, mm)
                .map_err(|e| invalid(format!("mm serialisation failed: {e}"))),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RenderedTokens(token_ids=<{} tokens>, message_indices=<{} entries>, multi_modal_data={})",
            self.inner.token_ids.len(),
            self.inner.message_indices.len(),
            if self.inner.multi_modal_data.is_some() {
                "Some(...)"
            } else {
                "None"
            },
        )
    }
}

#[pyclass(name = "ParsedToolCall", module = "renderers_native")]
#[derive(Clone)]
struct PyParsedToolCall {
    inner: ParsedToolCall,
}

#[pymethods]
impl PyParsedToolCall {
    #[getter]
    fn raw(&self) -> &str {
        &self.inner.raw
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        self.inner.name.as_deref()
    }

    #[getter]
    fn arguments<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner.arguments {
            None => Ok(py.None().into_bound(py)),
            Some(ToolArguments::Object(v)) => pythonize::pythonize(py, v)
                .map_err(|e| invalid(format!("args serialisation: {e}"))),
            Some(ToolArguments::Raw(s)) => Ok(s.clone().into_py(py).into_bound(py)),
        }
    }

    #[getter]
    fn token_span(&self) -> Option<(usize, usize)> {
        self.inner.token_span.as_ref().map(|r| (r.start, r.end))
    }

    #[getter]
    fn status(&self) -> &'static str {
        self.inner.status.as_wire()
    }

    #[getter]
    fn id(&self) -> Option<&str> {
        self.inner.id.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "ParsedToolCall(name={:?}, status={:?}, has_args={})",
            self.inner.name,
            self.inner.status,
            self.inner.arguments.is_some(),
        )
    }
}

#[pyclass(name = "ParsedResponse", module = "renderers_native")]
#[derive(Clone)]
struct PyParsedResponse {
    inner: ParsedResponse,
}

#[pymethods]
impl PyParsedResponse {
    #[getter]
    fn content(&self) -> &str {
        &self.inner.content
    }

    #[getter]
    fn reasoning_content(&self) -> Option<&str> {
        self.inner.reasoning_content.as_deref()
    }

    #[getter]
    fn tool_calls(&self) -> Vec<PyParsedToolCall> {
        self.inner
            .tool_calls
            .iter()
            .cloned()
            .map(|c| PyParsedToolCall { inner: c })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ParsedResponse(content_len={}, reasoning_content={}, tool_calls={})",
            self.inner.content.len(),
            self.inner.reasoning_content.is_some(),
            self.inner.tool_calls.len(),
        )
    }
}

/// Wire enum mirror — matches the Python `ToolCallParseStatus` string
/// values so existing code reading `tc.status == "ok"` keeps working.
#[pyclass(name = "ToolCallParseStatus", module = "renderers_native")]
#[derive(Clone, Copy)]
struct PyToolCallParseStatus {
    inner: ToolCallParseStatus,
}

#[pymethods]
impl PyToolCallParseStatus {
    #[classattr]
    const OK: &'static str = "ok";
    #[classattr]
    const INVALID_JSON: &'static str = "invalid_json";
    #[classattr]
    const UNCLOSED_BLOCK: &'static str = "unclosed_block";
    #[classattr]
    const MISSING_NAME: &'static str = "missing_name";
    #[classattr]
    const MALFORMED_STRUCTURE: &'static str = "malformed_structure";

    #[getter]
    fn value(&self) -> &'static str {
        self.inner.as_wire()
    }
}

/// Polymorphic Python-facing renderer.
#[pyclass(name = "Renderer", module = "renderers_native")]
struct PyRenderer {
    inner: Arc<dyn CoreRenderer>,
}

#[pymethods]
impl PyRenderer {
    /// Construct a Qwen3 renderer from a tokenizer.json on disk.
    ///
    /// Kept as an explicit classmethod (rather than `__new__`) so the
    /// type signature stays unambiguous from Python and future families
    /// can add their own classmethods.
    #[classmethod]
    #[pyo3(signature = (
        tokenizer_path,
        *,
        enable_thinking = true,
        preserve_all_thinking = false,
        preserve_thinking_between_tool_calls = false,
    ))]
    fn qwen3(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        tokenizer_path: &str,
        enable_thinking: bool,
        preserve_all_thinking: bool,
        preserve_thinking_between_tool_calls: bool,
    ) -> PyResult<Self> {
        let tok = Tokenizer::from_file(tokenizer_path).map_err(render_err)?;
        let renderer = py
            .allow_threads(|| {
                Qwen3RendererBuilder::default()
                    .enable_thinking(enable_thinking)
                    .preserve_all_thinking(preserve_all_thinking)
                    .preserve_thinking_between_tool_calls(preserve_thinking_between_tool_calls)
                    .build(tok)
            })
            .map_err(render_err)?;
        Ok(PyRenderer {
            inner: Arc::new(renderer),
        })
    }

    /// Build a Qwen3.5 renderer (text-only path) from a tokenizer.json.
    ///
    /// `enable_thinking` defaults to `True` (big-size variant). The Python
    /// shim is expected to probe the tokenizer's Jinja template to pick
    /// the right polarity for 0.8B / 2B models and forward it explicitly.
    #[classmethod]
    #[pyo3(signature = (
        tokenizer_path,
        *,
        enable_thinking = true,
        preserve_all_thinking = false,
        preserve_thinking_between_tool_calls = false,
    ))]
    fn qwen35(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        tokenizer_path: &str,
        enable_thinking: bool,
        preserve_all_thinking: bool,
        preserve_thinking_between_tool_calls: bool,
    ) -> PyResult<Self> {
        let tok = Tokenizer::from_file(tokenizer_path).map_err(render_err)?;
        let renderer = py
            .allow_threads(|| {
                Qwen35RendererBuilder::default()
                    .enable_thinking(enable_thinking)
                    .preserve_all_thinking(preserve_all_thinking)
                    .preserve_thinking_between_tool_calls(preserve_thinking_between_tool_calls)
                    .build(tok)
            })
            .map_err(render_err)?;
        Ok(PyRenderer {
            inner: Arc::new(renderer),
        })
    }

    /// Build a Qwen3.6 renderer (Qwen3.5 + JSON-flavoured tool args).
    #[classmethod]
    #[pyo3(signature = (
        tokenizer_path,
        *,
        enable_thinking = true,
        preserve_all_thinking = false,
        preserve_thinking_between_tool_calls = false,
    ))]
    fn qwen36(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        tokenizer_path: &str,
        enable_thinking: bool,
        preserve_all_thinking: bool,
        preserve_thinking_between_tool_calls: bool,
    ) -> PyResult<Self> {
        let tok = Tokenizer::from_file(tokenizer_path).map_err(render_err)?;
        let renderer = py
            .allow_threads(|| {
                Qwen36RendererBuilder::default()
                    .enable_thinking(enable_thinking)
                    .preserve_all_thinking(preserve_all_thinking)
                    .preserve_thinking_between_tool_calls(preserve_thinking_between_tool_calls)
                    .build(tok)
            })
            .map_err(render_err)?;
        Ok(PyRenderer {
            inner: Arc::new(renderer),
        })
    }

    /// Build a GLM-5 renderer from a tokenizer.json.
    #[classmethod]
    #[pyo3(signature = (
        tokenizer_path,
        *,
        enable_thinking = true,
        preserve_all_thinking = false,
        preserve_thinking_between_tool_calls = false,
    ))]
    fn glm5(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        tokenizer_path: &str,
        enable_thinking: bool,
        preserve_all_thinking: bool,
        preserve_thinking_between_tool_calls: bool,
    ) -> PyResult<Self> {
        let tok = Tokenizer::from_file(tokenizer_path).map_err(render_err)?;
        let renderer = py
            .allow_threads(|| {
                GlmRendererBuilder::glm5()
                    .enable_thinking(enable_thinking)
                    .preserve_all_thinking(preserve_all_thinking)
                    .preserve_thinking_between_tool_calls(preserve_thinking_between_tool_calls)
                    .build(tok)
            })
            .map_err(render_err)?;
        Ok(PyRenderer { inner: Arc::new(renderer) })
    }

    /// Build a GLM-5.1 renderer (GLM-5 + empty <think></think> on last assistant).
    #[classmethod]
    #[pyo3(signature = (
        tokenizer_path,
        *,
        enable_thinking = true,
        preserve_all_thinking = false,
        preserve_thinking_between_tool_calls = false,
    ))]
    fn glm51(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        tokenizer_path: &str,
        enable_thinking: bool,
        preserve_all_thinking: bool,
        preserve_thinking_between_tool_calls: bool,
    ) -> PyResult<Self> {
        let tok = Tokenizer::from_file(tokenizer_path).map_err(render_err)?;
        let renderer = py
            .allow_threads(|| {
                GlmRendererBuilder::glm51()
                    .enable_thinking(enable_thinking)
                    .preserve_all_thinking(preserve_all_thinking)
                    .preserve_thinking_between_tool_calls(preserve_thinking_between_tool_calls)
                    .build(tok)
            })
            .map_err(render_err)?;
        Ok(PyRenderer { inner: Arc::new(renderer) })
    }

    /// Build a GLM-4.5 Air renderer from a tokenizer.json.
    #[classmethod]
    #[pyo3(signature = (
        tokenizer_path,
        *,
        enable_thinking = true,
        preserve_all_thinking = false,
        preserve_thinking_between_tool_calls = false,
    ))]
    fn glm45(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        tokenizer_path: &str,
        enable_thinking: bool,
        preserve_all_thinking: bool,
        preserve_thinking_between_tool_calls: bool,
    ) -> PyResult<Self> {
        let tok = Tokenizer::from_file(tokenizer_path).map_err(render_err)?;
        let renderer = py
            .allow_threads(|| {
                GlmRendererBuilder::glm45()
                    .enable_thinking(enable_thinking)
                    .preserve_all_thinking(preserve_all_thinking)
                    .preserve_thinking_between_tool_calls(preserve_thinking_between_tool_calls)
                    .build(tok)
            })
            .map_err(render_err)?;
        Ok(PyRenderer { inner: Arc::new(renderer) })
    }

    /// Build a Nemotron 3 renderer from a tokenizer.json.
    ///
    /// `<|endoftext|>` is auto-detected: Nemotron-3 Nano / Super ship
    /// with only `<|im_end|>` as EOS; larger variants add `<|endoftext|>`.
    #[classmethod]
    #[pyo3(signature = (
        tokenizer_path,
        *,
        enable_thinking = true,
        preserve_all_thinking = false,
        preserve_thinking_between_tool_calls = false,
    ))]
    fn nemotron3(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        tokenizer_path: &str,
        enable_thinking: bool,
        preserve_all_thinking: bool,
        preserve_thinking_between_tool_calls: bool,
    ) -> PyResult<Self> {
        let tok = Tokenizer::from_file(tokenizer_path).map_err(render_err)?;
        let renderer = py
            .allow_threads(|| {
                Nemotron3RendererBuilder::default()
                    .enable_thinking(enable_thinking)
                    .preserve_all_thinking(preserve_all_thinking)
                    .preserve_thinking_between_tool_calls(preserve_thinking_between_tool_calls)
                    .build(tok)
            })
            .map_err(render_err)?;
        Ok(PyRenderer {
            inner: Arc::new(renderer),
        })
    }

    /// Build a DeepSeek V3 renderer from a tokenizer.json.
    ///
    /// `enable_thinking=True` (default) prefills the generation prompt
    /// with `<think>\n` to trigger reasoning. The Python shim mirrors
    /// the upstream class signature.
    #[classmethod]
    #[pyo3(signature = (tokenizer_path, *, enable_thinking = true))]
    fn deepseek_v3(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        tokenizer_path: &str,
        enable_thinking: bool,
    ) -> PyResult<Self> {
        let tok = Tokenizer::from_file(tokenizer_path).map_err(render_err)?;
        let renderer = py
            .allow_threads(|| {
                DeepSeekV3RendererBuilder::default()
                    .enable_thinking(enable_thinking)
                    .build(tok)
            })
            .map_err(render_err)?;
        Ok(PyRenderer {
            inner: Arc::new(renderer),
        })
    }

    #[pyo3(signature = (messages, *, tools = None, add_generation_prompt = false))]
    fn render(
        &self,
        py: Python<'_>,
        messages: &Bound<'_, PyAny>,
        tools: Option<&Bound<'_, PyAny>>,
        add_generation_prompt: bool,
    ) -> PyResult<PyRenderedTokens> {
        let msgs = parse_messages(messages)?;
        let tools = parse_tools(tools)?;
        let renderer = self.inner.clone();
        let out = py
            .allow_threads(move || renderer.render(&msgs, tools.as_deref(), add_generation_prompt))
            .map_err(render_err)?;
        Ok(PyRenderedTokens { inner: out })
    }

    #[pyo3(signature = (messages, *, tools = None, add_generation_prompt = false))]
    fn render_ids<'py>(
        &self,
        py: Python<'py>,
        messages: &Bound<'_, PyAny>,
        tools: Option<&Bound<'_, PyAny>>,
        add_generation_prompt: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let msgs = parse_messages(messages)?;
        let tools = parse_tools(tools)?;
        let renderer = self.inner.clone();
        let ids = py
            .allow_threads(move || {
                renderer.render_ids(&msgs, tools.as_deref(), add_generation_prompt)
            })
            .map_err(render_err)?;
        Ok(PyList::new_bound(py, ids.iter().map(|&t| t as i64)))
    }

    fn parse_response(
        &self,
        py: Python<'_>,
        token_ids: &Bound<'_, PyAny>,
    ) -> PyResult<PyParsedResponse> {
        let ids = parse_u32_list(token_ids)?;
        let renderer = self.inner.clone();
        let parsed = py.allow_threads(move || renderer.parse_response(&ids));
        Ok(PyParsedResponse { inner: parsed })
    }

    fn get_stop_token_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::new_bound(py, self.inner.stop_token_ids().iter().map(|&t| t as i64))
    }

    #[pyo3(signature = (previous_prompt_ids, previous_completion_ids, new_messages, *, tools = None))]
    fn bridge_to_next_turn(
        &self,
        py: Python<'_>,
        previous_prompt_ids: &Bound<'_, PyAny>,
        previous_completion_ids: &Bound<'_, PyAny>,
        new_messages: &Bound<'_, PyAny>,
        tools: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<PyRenderedTokens>> {
        let prev_p = parse_u32_list(previous_prompt_ids)?;
        let prev_c = parse_u32_list(previous_completion_ids)?;
        let msgs = parse_messages(new_messages)?;
        let tools = parse_tools(tools)?;
        let renderer = self.inner.clone();
        let bridged = py
            .allow_threads(move || {
                renderer.bridge_to_next_turn(&prev_p, &prev_c, &msgs, tools.as_deref())
            })
            .map_err(render_err)?;
        Ok(bridged.map(|rt| PyRenderedTokens { inner: rt }))
    }
}

#[pymodule]
fn renderers_native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_class::<PyRenderer>()?;
    m.add_class::<PyRenderedTokens>()?;
    m.add_class::<PyParsedResponse>()?;
    m.add_class::<PyParsedToolCall>()?;
    m.add_class::<PyToolCallParseStatus>()?;
    Ok(())
}
