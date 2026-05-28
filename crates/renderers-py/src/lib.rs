//! Python bindings for `renderers-core`.
//!
//! The boundary is intentionally thin: one polymorphic `Renderer`
//! pyclass holds an `Arc<dyn renderers_core::Renderer>`; small result
//! pyclasses wrap `RenderedTokens` / `ParsedResponse` / `ParsedToolCall`
//! with `#[getter]` accessors. Argument unpacking is done by
//! `pythonize` so callers can pass plain dicts / lists for messages and
//! tools without per-field `PyO3` conversion.

use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};
use rayon::prelude::*;

use renderers_core::Renderer as CoreRenderer;
use renderers_core::families::{
    DeepSeekV3RendererBuilder, DefaultRendererBuilder, GlmRendererBuilder, GptOssRendererBuilder,
    KimiK2RendererBuilder, KimiK25RendererBuilder, MiniMaxM2RendererBuilder,
    Nemotron3RendererBuilder, Qwen3RendererBuilder, Qwen35RendererBuilder, Qwen36RendererBuilder,
};
use renderers_core::processing::{ProcessedImage, Qwen3VlImageProcessor};
use renderers_core::tokenizer::Tokenizer;
use renderers_core::types::{
    Content, Message, ParsedResponse, ParsedToolCall, RenderedTokens, ToolArguments,
    ToolCallParseStatus, ToolSpec,
};
use renderers_core::types::{MediaBundle, MediaItem, Modality};

// Kept by-value so call sites can use the bare fn pointer
// `.map_err(render_err)` (closures would be needed for `&E`).
#[allow(clippy::needless_pass_by_value)]
fn render_err(e: renderers_core::types::RenderError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn invalid(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Decode a Python `list[dict]` of messages.
///
/// The hot path is plain OpenAI-style dictionaries with string fields.
/// Hand-parsing that shape avoids routing every render through generic
/// serde conversion while still falling back to `pythonize` for structured
/// content parts and tool-call lists.
fn parse_messages(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Message>> {
    let list = obj
        .cast::<PyList>()
        .map_err(|_| invalid("messages must be a list of dicts"))?;
    let mut parsed = Vec::with_capacity(list.len());
    for item in list.iter() {
        let dict = item
            .cast::<PyDict>()
            .map_err(|_| invalid("messages must be a list of dicts"))?;
        let role = dict
            .get_item("role")?
            .ok_or_else(|| invalid("message missing role"))?
            .extract::<String>()?;

        let content = match dict.get_item("content")? {
            None => Content::default(),
            Some(value) if value.is_none() => Content::Null,
            Some(value) => match value.extract::<String>() {
                Ok(text) => Content::Text(text),
                Err(_) => pythonize::depythonize(&value)
                    .map_err(|e| invalid(format!("message content decode failed: {e}")))?,
            },
        };

        let tool_calls = match dict.get_item("tool_calls")? {
            None => Vec::new(),
            Some(value) if value.is_none() => Vec::new(),
            Some(value) => pythonize::depythonize(&value)
                .map_err(|e| invalid(format!("message tool_calls decode failed: {e}")))?,
        };
        let tool_call_id = optional_string(dict, "tool_call_id")?;
        let name = optional_string(dict, "name")?;
        let reasoning_content = optional_string(dict, "reasoning_content")?;

        parsed.push(Message {
            role,
            content,
            tool_calls,
            tool_call_id,
            name,
            reasoning_content,
        });
    }
    Ok(parsed)
}

fn optional_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    match dict.get_item(key)? {
        None => Ok(None),
        Some(value) if value.is_none() => Ok(None),
        Some(value) => value.extract::<String>().map(Some),
    }
}

fn parse_tools(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Arc<Vec<ToolSpec>>>> {
    let Some(obj) = obj else { return Ok(None) };
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(prepared) = obj.extract::<PyRef<'_, PyPreparedTools>>() {
        return Ok(Some(prepared.inner.clone()));
    }
    let list = obj
        .cast::<PyList>()
        .map_err(|_| invalid("tools must be a list of dicts or PreparedTools"))?;
    let mut parsed = Vec::with_capacity(list.len());
    for item in list.iter() {
        let dict = item
            .cast::<PyDict>()
            .map_err(|_| invalid("tools must be a list of dicts"))?;
        let mut openai_envelope = false;
        let spec = if let Some(function) = dict.get_item("function")? {
            if let Ok(function_dict) = function.cast::<PyDict>() {
                openai_envelope = true;
                function_dict.clone()
            } else {
                dict.clone()
            }
        } else {
            dict.clone()
        };
        let name = spec
            .get_item("name")?
            .ok_or_else(|| invalid("tool spec missing name"))?
            .extract::<String>()?;
        let description = match spec.get_item("description")? {
            Some(value) => value.extract::<String>()?,
            None => String::new(),
        };
        let parameters = match spec.get_item("parameters")? {
            Some(value) => pythonize::depythonize(&value)
                .map_err(|e| invalid(format!("tool parameters decode failed: {e}")))?,
            None => serde_json::Value::Object(serde_json::Map::new()),
        };
        parsed.push(ToolSpec {
            name,
            description,
            parameters,
            openai_envelope,
        });
    }
    Ok(Some(Arc::new(parsed)))
}

#[inline]
fn tools_slice(tools: Option<&Arc<Vec<ToolSpec>>>) -> Option<&[ToolSpec]> {
    tools.map(|tools| tools.as_slice())
}

fn parse_message_batch(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Message>>> {
    let list = obj
        .cast::<PyList>()
        .map_err(|_| invalid("messages_batch must be a list of message lists"))?;
    let mut parsed = Vec::with_capacity(list.len());
    for item in list.iter() {
        parsed.push(parse_messages(&item)?);
    }
    Ok(parsed)
}

fn parse_fast_messages(
    roles: &Bound<'_, PyAny>,
    contents: &Bound<'_, PyAny>,
) -> PyResult<Vec<Message>> {
    let roles = roles
        .cast::<PyList>()
        .map_err(|_| invalid("roles must be a list[str]"))?;
    let contents = contents
        .cast::<PyList>()
        .map_err(|_| invalid("contents must be a list[str]"))?;
    if roles.len() != contents.len() {
        return Err(invalid("roles and contents must have the same length"));
    }
    let mut parsed = Vec::with_capacity(roles.len());
    for (role, content) in roles.iter().zip(contents.iter()) {
        parsed.push(Message {
            role: role.extract::<String>()?,
            content: Content::Text(content.extract::<String>()?),
            ..Default::default()
        });
    }
    Ok(parsed)
}

/// Decode a Python list of media-item dicts into a [`MediaBundle`].
fn parse_media_bundle(obj: &Bound<'_, PyAny>) -> PyResult<MediaBundle> {
    let value: serde_json::Value = pythonize::depythonize(obj)
        .map_err(|e| invalid(format!("media must be a list of dicts: {e}")))?;
    let serde_json::Value::Array(arr) = value else {
        return Err(invalid("media must be a list"));
    };
    let mut bundle = MediaBundle::new();
    for item in arr {
        let obj = item
            .as_object()
            .ok_or_else(|| invalid("media item must be a dict"))?;
        let message_idx =
            obj.get("message_idx")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| invalid("media item missing message_idx"))? as usize;
        let modality_str = obj
            .get("modality")
            .and_then(|v| v.as_str())
            .ok_or_else(|| invalid("media item missing modality"))?;
        let modality = match modality_str {
            "image" => Modality::Image,
            "video" => Modality::Video,
            other => return Err(invalid(format!("unknown modality: {other}"))),
        };
        let num_tokens =
            obj.get("num_tokens")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| invalid("media item missing num_tokens"))? as usize;
        let hash = obj
            .get("hash")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_default();
        let hf_payload = obj
            .get("hf_payload")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        bundle.push(
            message_idx,
            MediaItem {
                modality,
                hash,
                num_tokens,
                hf_payload,
            },
        );
    }
    Ok(bundle)
}

fn parse_u32_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    // Accept either a Python list of ints or a numpy-style sequence.
    let list = obj
        .cast::<PyList>()
        .map_err(|_| invalid("expected list[int]"))?;
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        out.push(item.extract::<u32>()?);
    }
    Ok(out)
}

fn numpy_u32_slice<'py>(array: &'py PyReadonlyArray1<'py, u32>) -> PyResult<&'py [u32]> {
    array
        .as_slice()
        .map_err(|e| invalid(format!("expected a contiguous uint32 numpy array: {e}")))
}

fn batch_ids_to_pylist(py: Python<'_>, batch_ids: Vec<Vec<u32>>) -> PyResult<Bound<'_, PyList>> {
    let mut rows = Vec::with_capacity(batch_ids.len());
    for ids in batch_ids {
        rows.push(PyList::new(py, ids)?);
    }
    PyList::new(py, rows)
}

#[pyclass(
    name = "RenderedTokens",
    module = "renderers_native",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyRenderedTokens {
    inner: RenderedTokens,
}

#[pymethods]
impl PyRenderedTokens {
    #[getter]
    fn token_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, &self.inner.token_ids)
    }

    #[getter]
    fn message_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.inner.message_indices.iter().copied())
    }

    #[getter]
    #[allow(clippy::unused_self)]
    fn sampled_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::empty(py)
    }

    #[getter]
    #[allow(clippy::unused_self)]
    fn is_content<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::empty(py)
    }

    #[getter]
    #[allow(clippy::unused_self)]
    fn message_roles<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::empty(py)
    }

    #[getter]
    fn multi_modal_data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner.multi_modal_data {
            Some(mm) => pythonize::pythonize(py, mm)
                .map_err(|e| invalid(format!("mm serialisation failed: {e}"))),
            None => Ok(py.None().into_bound(py)),
        }
    }

    #[pyo3(signature = (n_messages = None, *, sampled_only = false))]
    fn tokens_per_message<'py>(
        &self,
        py: Python<'py>,
        n_messages: Option<usize>,
        sampled_only: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let n_messages = n_messages.unwrap_or(0);
        let out = if sampled_only {
            vec![0usize; n_messages]
        } else {
            let mut counts = vec![0usize; n_messages];
            for idx in &self.inner.message_indices {
                let Ok(msg_idx) = usize::try_from(*idx) else {
                    continue;
                };
                if msg_idx < n_messages {
                    counts[msg_idx] += 1;
                }
            }
            counts
        };
        PyList::new(py, out)
    }

    fn message_token_spans<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let n_messages = self
            .inner
            .message_indices
            .iter()
            .copied()
            .filter(|idx| *idx >= 0)
            .max()
            .map_or(0usize, |idx| usize::try_from(idx).map_or(0, |idx| idx + 1));
        let mut firsts = vec![None::<usize>; n_messages];
        let mut lasts = vec![None::<usize>; n_messages];
        for (pos, idx) in self.inner.message_indices.iter().copied().enumerate() {
            let Ok(msg_idx) = usize::try_from(idx) else {
                continue;
            };
            if msg_idx >= n_messages {
                continue;
            }
            if firsts[msg_idx].is_none() {
                firsts[msg_idx] = Some(pos);
            }
            lasts[msg_idx] = Some(pos);
        }

        let out = PyList::empty(py);
        for (first, last) in firsts.into_iter().zip(lasts) {
            match (first, last) {
                (Some(start), Some(end)) => {
                    out.append(PyTuple::new(py, [start, end + 1])?)?;
                }
                _ => out.append(py.None())?,
            }
        }
        Ok(out)
    }

    #[allow(clippy::unused_self)]
    fn role_token_spans<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        PyDict::new(py)
    }

    #[pyo3(signature = (*, sampled_only = false))]
    #[allow(clippy::unused_self)]
    fn tokens_by_role<'py>(
        &self,
        py: Python<'py>,
        #[allow(unused_variables)] sampled_only: bool,
    ) -> Bound<'py, PyDict> {
        PyDict::new(py)
    }

    #[allow(clippy::unused_self)]
    fn content_token_spans_by_role<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        PyDict::new(py)
    }

    fn content_mask_for_roles<'py>(
        &self,
        py: Python<'py>,
        #[allow(unused_variables)] roles: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, vec![false; self.inner.token_ids.len()])
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

#[pyclass(
    name = "ParsedToolCall",
    module = "renderers_native",
    skip_from_py_object
)]
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
            Some(ToolArguments::Object(v)) => {
                pythonize::pythonize(py, v).map_err(|e| invalid(format!("args serialisation: {e}")))
            }
            Some(ToolArguments::Raw(s)) => Ok(s
                .as_str()
                .into_pyobject(py)
                .map_err(|e| invalid(format!("string into pyobject: {e}")))?
                .into_any()),
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

#[pyclass(
    name = "ParsedResponse",
    module = "renderers_native",
    skip_from_py_object
)]
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
#[pyclass(
    name = "ToolCallParseStatus",
    module = "renderers_native",
    skip_from_py_object
)]
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

    // PyO3 #[getter] requires `&self`; the Copy enum is 1 byte so clippy
    // suggests by-value, but the macro shape is fixed.
    #[getter]
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn value(&self) -> &'static str {
        self.inner.as_wire()
    }
}

#[pyclass(
    name = "PreparedTools",
    module = "renderers_native",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyPreparedTools {
    inner: Arc<Vec<ToolSpec>>,
}

#[pymethods]
impl PyPreparedTools {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("PreparedTools(<{} tools>)", self.inner.len())
    }
}

#[pyclass(name = "RendererSession", module = "renderers_native")]
struct PyRendererSession {
    renderer: Arc<dyn CoreRenderer>,
    messages: Arc<Vec<Message>>,
    tools: Option<Arc<Vec<ToolSpec>>>,
    last_prompt_ids: Option<Vec<u32>>,
}

#[pymethods]
impl PyRendererSession {
    fn fork(&self) -> Self {
        Self {
            renderer: self.renderer.clone(),
            messages: self.messages.clone(),
            tools: self.tools.clone(),
            last_prompt_ids: self.last_prompt_ids.clone(),
        }
    }

    #[pyo3(signature = (*, add_generation_prompt = false))]
    fn render_ids<'py>(
        &mut self,
        py: Python<'py>,
        add_generation_prompt: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let renderer = self.renderer.clone();
        let messages = self.messages.clone();
        let tools = self.tools.clone();
        let ids = py
            .detach(move || {
                renderer.render_ids(
                    messages.as_slice(),
                    tools_slice(tools.as_ref()),
                    add_generation_prompt,
                )
            })
            .map_err(render_err)?;
        let out = PyList::new(py, ids.iter().copied())?;
        self.last_prompt_ids = Some(ids);
        Ok(out)
    }

    #[pyo3(signature = (*, add_generation_prompt = false))]
    fn render_ids_np<'py>(
        &mut self,
        py: Python<'py>,
        add_generation_prompt: bool,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let renderer = self.renderer.clone();
        let messages = self.messages.clone();
        let tools = self.tools.clone();
        let ids = py
            .detach(move || {
                renderer.render_ids(
                    messages.as_slice(),
                    tools_slice(tools.as_ref()),
                    add_generation_prompt,
                )
            })
            .map_err(render_err)?;
        self.last_prompt_ids = Some(ids.clone());
        Ok(ids.into_pyarray(py))
    }

    #[pyo3(signature = (previous_completion_ids, new_messages, *, update = true))]
    fn bridge_to_next_turn(
        &mut self,
        py: Python<'_>,
        previous_completion_ids: &Bound<'_, PyAny>,
        new_messages: &Bound<'_, PyAny>,
        update: bool,
    ) -> PyResult<Option<PyRenderedTokens>> {
        let prev_p = self
            .last_prompt_ids
            .clone()
            .ok_or_else(|| invalid("render_ids must be called before session bridge"))?;
        let prev_c = parse_u32_list(previous_completion_ids)?;
        let msgs = parse_messages(new_messages)?;
        let renderer = self.renderer.clone();
        let tools = self.tools.clone();
        let bridged = py
            .detach(move || {
                renderer.bridge_to_next_turn(&prev_p, &prev_c, &msgs, tools_slice(tools.as_ref()))
            })
            .map_err(render_err)?;
        if update && let Some(rendered) = &bridged {
            self.last_prompt_ids = Some(rendered.token_ids.clone());
        }
        Ok(bridged.map(|rt| PyRenderedTokens { inner: rt }))
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (previous_completion_ids, new_messages, *, update = true))]
    fn bridge_to_next_turn_np<'py>(
        &mut self,
        py: Python<'py>,
        previous_completion_ids: PyReadonlyArray1<'_, u32>,
        new_messages: &Bound<'_, PyAny>,
        update: bool,
    ) -> PyResult<Option<Bound<'py, PyArray1<u32>>>> {
        let prev_p = self
            .last_prompt_ids
            .as_deref()
            .ok_or_else(|| invalid("render_ids must be called before session bridge"))?;
        let prev_c = numpy_u32_slice(&previous_completion_ids)?;
        let msgs = parse_messages(new_messages)?;
        let bridged = self
            .renderer
            .bridge_to_next_turn(prev_p, prev_c, &msgs, tools_slice(self.tools.as_ref()))
            .map_err(render_err)?;
        if let Some(rendered) = bridged {
            if update {
                self.last_prompt_ids = Some(rendered.token_ids.clone());
            }
            Ok(Some(rendered.token_ids.into_pyarray(py)))
        } else {
            Ok(None)
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RendererSession(messages={}, tools={}, has_prompt={})",
            self.messages.len(),
            self.tools.as_ref().map_or(0, |t| t.len()),
            self.last_prompt_ids.is_some(),
        )
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
            .detach(|| {
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

    /// Build a Qwen3-VL renderer — alias for [`Renderer.qwen35`].
    ///
    /// Qwen3-VL and Qwen3.5-VL share the same chat template and the
    /// same set of special tokens, so the renderer implementation is
    /// identical. The factory is exposed separately so callers reading
    /// from a registry can spell the family name directly.
    #[classmethod]
    #[pyo3(signature = (
        tokenizer_path,
        *,
        enable_thinking = true,
        preserve_all_thinking = false,
        preserve_thinking_between_tool_calls = false,
    ))]
    fn qwen3_vl(
        cls: &Bound<'_, PyType>,
        py: Python<'_>,
        tokenizer_path: &str,
        enable_thinking: bool,
        preserve_all_thinking: bool,
        preserve_thinking_between_tool_calls: bool,
    ) -> PyResult<Self> {
        Self::qwen35(
            cls,
            py,
            tokenizer_path,
            enable_thinking,
            preserve_all_thinking,
            preserve_thinking_between_tool_calls,
        )
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
            .detach(|| {
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
            .detach(|| {
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
            .detach(|| {
                GlmRendererBuilder::glm5()
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
            .detach(|| {
                GlmRendererBuilder::glm51()
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
            .detach(|| {
                GlmRendererBuilder::glm45()
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

    /// Build a `MiniMax` M2 / M2.5 renderer from a tokenizer.json.
    #[classmethod]
    #[pyo3(signature = (
        tokenizer_path,
        *,
        preserve_all_thinking = false,
        preserve_thinking_between_tool_calls = false,
    ))]
    fn minimax_m2(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        tokenizer_path: &str,
        preserve_all_thinking: bool,
        preserve_thinking_between_tool_calls: bool,
    ) -> PyResult<Self> {
        let tok = Tokenizer::from_file(tokenizer_path).map_err(render_err)?;
        let renderer = py
            .detach(|| {
                MiniMaxM2RendererBuilder::default()
                    .preserve_all_thinking(preserve_all_thinking)
                    .preserve_thinking_between_tool_calls(preserve_thinking_between_tool_calls)
                    .build(tok)
            })
            .map_err(render_err)?;
        Ok(PyRenderer {
            inner: Arc::new(renderer),
        })
    }

    /// Build a `DefaultRenderer` (Jinja fallback via minijinja).
    ///
    /// `chat_template` is the model's Jinja chat template (usually the
    /// `chat_template` field of `tokenizer_config.json` or the contents
    /// of `chat_template.jinja`). `stop_token_ids` is typically
    /// `[eos_token_id]`; pass `None` to leave it empty.
    #[classmethod]
    #[pyo3(signature = (tokenizer_path, chat_template, *, stop_token_ids = None, extra_context = None))]
    fn default_renderer(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        tokenizer_path: &str,
        chat_template: &str,
        stop_token_ids: Option<&Bound<'_, PyAny>>,
        extra_context: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let tok = Tokenizer::from_file(tokenizer_path).map_err(render_err)?;
        let stop_ids: Vec<u32> = match stop_token_ids {
            None => Vec::new(),
            Some(obj) if obj.is_none() => Vec::new(),
            Some(obj) => parse_u32_list(obj)?,
        };
        let extras: Vec<(String, serde_json::Value)> = match extra_context {
            None => Vec::new(),
            Some(obj) if obj.is_none() => Vec::new(),
            Some(obj) => {
                let v: serde_json::Value = pythonize::depythonize(obj)
                    .map_err(|e| invalid(format!("extra_context: {e}")))?;
                match v {
                    serde_json::Value::Object(m) => m.into_iter().collect(),
                    _ => return Err(invalid("extra_context must be a dict")),
                }
            }
        };
        let ct = chat_template.to_string();
        let renderer = py
            .detach(move || {
                let mut b = DefaultRendererBuilder::new(ct).stop_token_ids(stop_ids);
                for (k, v) in extras {
                    b = b.add_context(k, v);
                }
                b.build(tok)
            })
            .map_err(render_err)?;
        Ok(PyRenderer {
            inner: Arc::new(renderer),
        })
    }

    /// Build a GPT-OSS (Harmony) renderer.
    ///
    /// Unlike the other families, GPT-OSS doesn't need a `HuggingFace`
    /// `tokenizer.json` — the harmony encoding embeds its own
    /// tiktoken-based tokenizer. The `tokenizer_path` argument is
    /// ignored on this path but kept for API uniformity with the other
    /// classmethods (callers can pass an empty string).
    #[classmethod]
    #[pyo3(signature = (
        tokenizer_path,
        *,
        use_system_prompt = true,
        reasoning_effort = None,
        conversation_start_date = None,
        knowledge_cutoff = None,
        model_identity = None,
        preserve_all_thinking = false,
        preserve_thinking_between_tool_calls = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn gpt_oss(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        tokenizer_path: &str,
        use_system_prompt: bool,
        reasoning_effort: Option<&str>,
        conversation_start_date: Option<&str>,
        knowledge_cutoff: Option<&str>,
        model_identity: Option<&str>,
        preserve_all_thinking: bool,
        preserve_thinking_between_tool_calls: bool,
    ) -> PyResult<Self> {
        let _ = tokenizer_path; // not needed for harmony
        let effort = reasoning_effort.unwrap_or("medium").to_string();
        let renderer = py
            .detach(move || -> Result<_, renderers_core::types::RenderError> {
                let mut b = GptOssRendererBuilder::default()
                    .use_system_prompt(use_system_prompt)
                    .preserve_all_thinking(preserve_all_thinking)
                    .preserve_thinking_between_tool_calls(preserve_thinking_between_tool_calls);
                b = b.reasoning_effort(&effort)?;
                if let Some(d) = conversation_start_date {
                    b = b.conversation_start_date(d);
                }
                if let Some(k) = knowledge_cutoff {
                    b = b.knowledge_cutoff(k);
                }
                if let Some(m) = model_identity {
                    b = b.model_identity(m);
                }
                b.build()
            })
            .map_err(render_err)?;
        Ok(PyRenderer {
            inner: Arc::new(renderer),
        })
    }

    /// Build a Kimi K2.5 renderer (text-only, no tools).
    ///
    /// The Python shim is expected to route Kimi K2.5 to native ONLY
    /// when there are no tools and no image / video content — the
    /// TypeScript-style tool declaration formatter and the vision
    /// processor are still pure-Python in this phase.
    #[classmethod]
    #[pyo3(signature = (
        tokenizer_path,
        *,
        enable_thinking = true,
        preserve_all_thinking = false,
        preserve_thinking_between_tool_calls = false,
    ))]
    fn kimi_k25(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        tokenizer_path: &str,
        enable_thinking: bool,
        preserve_all_thinking: bool,
        preserve_thinking_between_tool_calls: bool,
    ) -> PyResult<Self> {
        let tok = Tokenizer::from_file(tokenizer_path).map_err(render_err)?;
        let renderer = py
            .detach(|| {
                KimiK25RendererBuilder::default()
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

    /// Build a Kimi K2 renderer from a tokenizer.json.
    #[classmethod]
    #[pyo3(signature = (
        tokenizer_path,
        *,
        enable_thinking = true,
        preserve_all_thinking = false,
        preserve_thinking_between_tool_calls = false,
    ))]
    fn kimi_k2(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        tokenizer_path: &str,
        enable_thinking: bool,
        preserve_all_thinking: bool,
        preserve_thinking_between_tool_calls: bool,
    ) -> PyResult<Self> {
        let tok = Tokenizer::from_file(tokenizer_path).map_err(render_err)?;
        let renderer = py
            .detach(|| {
                KimiK2RendererBuilder::default()
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
            .detach(|| {
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

    /// Build a `DeepSeek` V3 renderer from a tokenizer.json.
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
            .detach(|| {
                DeepSeekV3RendererBuilder::default()
                    .enable_thinking(enable_thinking)
                    .build(tok)
            })
            .map_err(render_err)?;
        Ok(PyRenderer {
            inner: Arc::new(renderer),
        })
    }

    #[allow(clippy::unused_self)]
    fn prepare_tools(&self, tools: &Bound<'_, PyAny>) -> PyResult<PyPreparedTools> {
        let parsed = parse_tools(Some(tools))?.unwrap_or_else(|| Arc::new(Vec::new()));
        Ok(PyPreparedTools { inner: parsed })
    }

    #[pyo3(signature = (messages, *, tools = None))]
    fn new_session(
        &self,
        messages: &Bound<'_, PyAny>,
        tools: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyRendererSession> {
        Ok(PyRendererSession {
            renderer: self.inner.clone(),
            messages: Arc::new(parse_messages(messages)?),
            tools: parse_tools(tools)?,
            last_prompt_ids: None,
        })
    }

    #[pyo3(signature = (messages_batch, *, tools = None, add_generation_prompt = false))]
    fn render_batch_ids<'py>(
        &self,
        py: Python<'py>,
        messages_batch: &Bound<'_, PyAny>,
        tools: Option<&Bound<'_, PyAny>>,
        add_generation_prompt: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let batch = parse_message_batch(messages_batch)?;
        let tools = parse_tools(tools)?;
        let renderer = self.inner.clone();
        let batch_ids = py
            .detach(move || {
                if batch.len() >= 8 {
                    batch
                        .par_iter()
                        .map(|messages| {
                            renderer.render_ids(
                                messages,
                                tools_slice(tools.as_ref()),
                                add_generation_prompt,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()
                } else {
                    batch
                        .iter()
                        .map(|messages| {
                            renderer.render_ids(
                                messages,
                                tools_slice(tools.as_ref()),
                                add_generation_prompt,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()
                }
            })
            .map_err(render_err)?;
        batch_ids_to_pylist(py, batch_ids)
    }

    #[pyo3(signature = (messages_batch, *, tools = None, add_generation_prompt = false))]
    fn render_batch_ids_np_packed<'py>(
        &self,
        py: Python<'py>,
        messages_batch: &Bound<'_, PyAny>,
        tools: Option<&Bound<'_, PyAny>>,
        add_generation_prompt: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let batch = parse_message_batch(messages_batch)?;
        let tools = parse_tools(tools)?;
        let renderer = self.inner.clone();
        let (ids, offsets) = py
            .detach(
                move || -> Result<(Vec<u32>, Vec<i64>), renderers_core::types::RenderError> {
                    let batch_ids = if batch.len() >= 8 {
                        batch
                            .par_iter()
                            .map(|messages| {
                                renderer.render_ids(
                                    messages,
                                    tools_slice(tools.as_ref()),
                                    add_generation_prompt,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        batch
                            .iter()
                            .map(|messages| {
                                renderer.render_ids(
                                    messages,
                                    tools_slice(tools.as_ref()),
                                    add_generation_prompt,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?
                    };
                    let total_len = batch_ids.iter().map(Vec::len).sum();
                    let mut ids = Vec::with_capacity(total_len);
                    let mut offsets = Vec::with_capacity(batch_ids.len() + 1);
                    offsets.push(0);
                    for row in batch_ids {
                        ids.extend_from_slice(&row);
                        offsets.push(ids.len() as i64);
                    }
                    Ok((ids, offsets))
                },
            )
            .map_err(render_err)?;
        let ids = ids.into_pyarray(py).into_any();
        let offsets = offsets.into_pyarray(py).into_any();
        PyTuple::new(py, [ids, offsets])
    }

    #[pyo3(signature = (roles, contents, *, tools = None, add_generation_prompt = false))]
    fn render_fast_ids<'py>(
        &self,
        py: Python<'py>,
        roles: &Bound<'_, PyAny>,
        contents: &Bound<'_, PyAny>,
        tools: Option<&Bound<'_, PyAny>>,
        add_generation_prompt: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let messages = parse_fast_messages(roles, contents)?;
        let tools = parse_tools(tools)?;
        let renderer = self.inner.clone();
        let ids = py
            .detach(move || {
                renderer.render_ids(
                    &messages,
                    tools_slice(tools.as_ref()),
                    add_generation_prompt,
                )
            })
            .map_err(render_err)?;
        PyList::new(py, ids)
    }

    #[pyo3(signature = (roles, contents, *, tools = None, add_generation_prompt = false))]
    fn render_fast_ids_np<'py>(
        &self,
        py: Python<'py>,
        roles: &Bound<'_, PyAny>,
        contents: &Bound<'_, PyAny>,
        tools: Option<&Bound<'_, PyAny>>,
        add_generation_prompt: bool,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let messages = parse_fast_messages(roles, contents)?;
        let tools = parse_tools(tools)?;
        let renderer = self.inner.clone();
        let ids = py
            .detach(move || {
                renderer.render_ids(
                    &messages,
                    tools_slice(tools.as_ref()),
                    add_generation_prompt,
                )
            })
            .map_err(render_err)?;
        Ok(ids.into_pyarray(py))
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
            .detach(move || {
                renderer.render(&msgs, tools_slice(tools.as_ref()), add_generation_prompt)
            })
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
            .detach(move || {
                renderer.render_ids(&msgs, tools_slice(tools.as_ref()), add_generation_prompt)
            })
            .map_err(render_err)?;
        PyList::new(py, ids)
    }

    /// Render token ids as a `numpy.ndarray[np.uint32]`.
    ///
    /// This transfers the Rust `Vec<u32>` allocation into `NumPy` instead of
    /// materialising a Python `list[int]`, which is the preferred hot-path
    /// API for benchmark loops and inference clients that already operate on
    /// array buffers.
    #[pyo3(signature = (messages, *, tools = None, add_generation_prompt = false))]
    fn render_ids_np<'py>(
        &self,
        py: Python<'py>,
        messages: &Bound<'_, PyAny>,
        tools: Option<&Bound<'_, PyAny>>,
        add_generation_prompt: bool,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let msgs = parse_messages(messages)?;
        let tools = parse_tools(tools)?;
        let renderer = self.inner.clone();
        let ids = py
            .detach(move || {
                renderer.render_ids(&msgs, tools_slice(tools.as_ref()), add_generation_prompt)
            })
            .map_err(render_err)?;
        Ok(ids.into_pyarray(py))
    }

    #[pyo3(signature = (token_ids, *, tools = None))]
    fn parse_response(
        &self,
        py: Python<'_>,
        token_ids: &Bound<'_, PyAny>,
        #[allow(unused_variables)] tools: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyParsedResponse> {
        let ids = parse_u32_list(token_ids)?;
        let renderer = self.inner.clone();
        let parsed = py.detach(move || renderer.parse_response(&ids));
        Ok(PyParsedResponse { inner: parsed })
    }

    /// Parse completion ids from a contiguous `numpy.ndarray[np.uint32]`.
    ///
    /// The input buffer is borrowed directly, avoiding the Python-list scan and
    /// temporary Rust `Vec<u32>` used by `parse_response`.
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (token_ids, *, tools = None))]
    fn parse_response_np(
        &self,
        token_ids: PyReadonlyArray1<'_, u32>,
        #[allow(unused_variables)] tools: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyParsedResponse> {
        let ids = numpy_u32_slice(&token_ids)?;
        let parsed = self.inner.parse_response(ids);
        Ok(PyParsedResponse { inner: parsed })
    }

    fn get_stop_token_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.inner.stop_token_ids())
    }

    /// Render with pre-resolved multimodal media items.
    ///
    /// `media` is a list of dicts each shaped like
    /// ``{"message_idx": int, "modality": "image" | "video",
    ///    "num_tokens": int, "hash": str, "hf_payload": <any>}``.
    /// `num_tokens` is the placeholder expansion count pre-computed by
    /// the caller's vision processor (HF
    /// ``image_grid_thw.prod()/merge_size**2`` for Qwen-VL). The Rust
    /// renderer never touches pixel data — `hf_payload` rides through
    /// as opaque JSON into `multi_modal_data.mm_items`.
    ///
    /// Raises ``RuntimeError`` when the underlying family doesn't
    /// support multimodal (e.g. a Qwen3.5 text-only tokenizer that
    /// doesn't ship the ``<|vision_start|>`` token).
    #[pyo3(signature = (messages, media, *, tools = None, add_generation_prompt = false))]
    fn render_with_media(
        &self,
        py: Python<'_>,
        messages: &Bound<'_, PyAny>,
        media: &Bound<'_, PyAny>,
        tools: Option<&Bound<'_, PyAny>>,
        add_generation_prompt: bool,
    ) -> PyResult<PyRenderedTokens> {
        let msgs = parse_messages(messages)?;
        let tools = parse_tools(tools)?;
        let bundle = parse_media_bundle(media)?;
        let renderer = self.inner.clone();
        let out = py
            .detach(move || -> Result<_, renderers_core::types::RenderError> {
                let mm = renderer
                    .as_multimodal()
                    .ok_or_else(|| renderers_core::types::RenderError::Invalid(
                        "this renderer does not support multimodal — use a -VL tokenizer or check supports_multimodal()".into(),
                    ))?;
                mm.render_with_media(
                    &msgs,
                    tools_slice(tools.as_ref()),
                    &bundle,
                    add_generation_prompt,
                )
            })
            .map_err(render_err)?;
        Ok(PyRenderedTokens { inner: out })
    }

    /// True when the underlying family supports the multimodal trait
    /// AND the loaded tokenizer ships the modality special tokens.
    fn supports_multimodal(&self) -> bool {
        self.inner.as_multimodal().is_some()
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
            .detach(move || {
                renderer.bridge_to_next_turn(&prev_p, &prev_c, &msgs, tools_slice(tools.as_ref()))
            })
            .map_err(render_err)?;
        Ok(bridged.map(|rt| PyRenderedTokens { inner: rt }))
    }

    /// Bridge using `NumPy` token buffers and return a `NumPy` token buffer.
    ///
    /// Previous prompt/completion ids are borrowed directly from contiguous
    /// `uint32` arrays, and the bridged Rust `Vec<u32>` is transferred into
    /// `NumPy` on output. This is the lowest-overhead Python-facing bridge path.
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (previous_prompt_ids, previous_completion_ids, new_messages, *, tools = None))]
    fn bridge_to_next_turn_np<'py>(
        &self,
        py: Python<'py>,
        previous_prompt_ids: PyReadonlyArray1<'_, u32>,
        previous_completion_ids: PyReadonlyArray1<'_, u32>,
        new_messages: &Bound<'_, PyAny>,
        tools: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyArray1<u32>>>> {
        let prev_p = numpy_u32_slice(&previous_prompt_ids)?;
        let prev_c = numpy_u32_slice(&previous_completion_ids)?;
        let msgs = parse_messages(new_messages)?;
        let tools = parse_tools(tools)?;
        let bridged = self
            .inner
            .bridge_to_next_turn(prev_p, prev_c, &msgs, tools_slice(tools.as_ref()))
            .map_err(render_err)?;
        Ok(bridged.map(|rt| rt.token_ids.into_pyarray(py)))
    }
}

// ── Vision: Qwen3-VL image processor ──────────────────────────────────

/// Rust port of HF's `Qwen3VLImageProcessor` / `Qwen2VLImageProcessor`.
///
/// Decodes image bytes, smart-resizes, normalises with the `OpenAI` CLIP
/// mean / std, and produces `pixel_values` + `image_grid_thw` tensors
/// in the exact shape the model expects. Equivalent to the Python
/// processor end-to-end; pixel-byte parity is approximate (`CatmullRom`
/// vs PIL bicubic), but grid dims, `num_tokens`, and tensor shape match
/// exactly.
#[pyclass(name = "Qwen3VlImageProcessor", module = "renderers_native")]
struct PyQwen3VlImageProcessor {
    inner: Qwen3VlImageProcessor,
}

#[pymethods]
impl PyQwen3VlImageProcessor {
    #[new]
    #[pyo3(signature = (
        *,
        min_pixels = None,
        max_pixels = None,
        patch_size = None,
        temporal_patch_size = None,
        merge_size = None,
    ))]
    fn new(
        min_pixels: Option<u32>,
        max_pixels: Option<u32>,
        patch_size: Option<u32>,
        temporal_patch_size: Option<u32>,
        merge_size: Option<u32>,
    ) -> Self {
        let mut p = Qwen3VlImageProcessor::default();
        if let Some(v) = min_pixels {
            p.min_pixels = v;
        }
        if let Some(v) = max_pixels {
            p.max_pixels = v;
        }
        if let Some(v) = patch_size {
            p.patch_size = v;
        }
        if let Some(v) = temporal_patch_size {
            p.temporal_patch_size = v;
        }
        if let Some(v) = merge_size {
            p.merge_size = v;
        }
        Self { inner: p }
    }

    /// Compute the resized `(height, width)` for an input image
    /// without doing any actual pixel work — useful for placeholder
    /// counting in test harnesses.
    fn smart_resize(&self, height: u32, width: u32) -> PyResult<(u32, u32)> {
        self.inner.smart_resize(height, width).map_err(render_err)
    }

    /// Process raw image bytes (PNG / JPEG / WebP) into a dict shaped
    /// for direct consumption by `Renderer.render_with_media`:
    ///
    /// ```python
    /// {
    ///     "modality":    "image",
    ///     "num_tokens":  int,
    ///     "hash":        str,
    ///     "hf_payload":  {
    ///         "pixel_values":   {"shape": [tokens, features], "data": [...]},
    ///         "image_grid_thw": {"shape": [1, 3],             "data": [1, h, w]},
    ///     },
    /// }
    /// ```
    ///
    /// `message_idx` is up to the caller — it's not added here.
    fn process_bytes<'py>(&self, py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        // Clone so the move into detach is straightforward
        let processed: ProcessedImage = py
            .detach(|| self.inner.process_bytes(bytes))
            .map_err(render_err)?;
        processed_to_pyobject(py, processed)
    }

    /// Convenience: read a file and process it.
    fn process_path<'py>(&self, py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyAny>> {
        let bytes =
            std::fs::read(path).map_err(|e| invalid(format!("read image {path:?}: {e}")))?;
        let processed: ProcessedImage = py
            .detach(|| self.inner.process_bytes(&bytes))
            .map_err(render_err)?;
        processed_to_pyobject(py, processed)
    }

    #[getter]
    fn patch_size(&self) -> u32 {
        self.inner.patch_size
    }
    #[getter]
    fn merge_size(&self) -> u32 {
        self.inner.merge_size
    }
    #[getter]
    fn temporal_patch_size(&self) -> u32 {
        self.inner.temporal_patch_size
    }
    #[getter]
    fn min_pixels(&self) -> u32 {
        self.inner.min_pixels
    }
    #[getter]
    fn max_pixels(&self) -> u32 {
        self.inner.max_pixels
    }
}

fn processed_to_pyobject<'py>(py: Python<'py>, p: ProcessedImage) -> PyResult<Bound<'py, PyAny>> {
    // Zero-copy: hand numpy the Vec<f32> directly. The numpy array
    // takes ownership of the buffer, so this avoids the per-element
    // PyFloat allocation that the previous nested-list path triggered.
    // Shape: (num_tokens × merge², 3 × temporal × patch²).
    let shape = (p.pixel_values.shape()[0], p.pixel_values.shape()[1]);
    let pixel_array: Bound<'py, PyArray2<f32>> = p.pixel_values.into_pyarray(py);
    let grid_array: Bound<'py, PyArray2<i64>> = ndarray::Array2::from_shape_vec(
        (1, 3),
        p.image_grid_thw.iter().copied().map(i64::from).collect(),
    )
    .expect("image_grid_thw is always shape [1,3]")
    .into_pyarray(py);

    let hf_payload = PyDict::new(py);
    hf_payload.set_item("pixel_values", pixel_array)?;
    hf_payload.set_item("image_grid_thw", grid_array)?;

    let out = PyDict::new(py);
    out.set_item("modality", "image")?;
    out.set_item("num_tokens", p.num_tokens)?;
    out.set_item("hash", p.hash)?;
    out.set_item("hf_payload", hf_payload)?;
    let _ = shape; // shape captured in the numpy array's own metadata
    Ok(out.into_any())
}

#[pymodule]
fn renderers_native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_class::<PyRenderer>()?;
    m.add_class::<PyPreparedTools>()?;
    m.add_class::<PyRendererSession>()?;
    m.add_class::<PyRenderedTokens>()?;
    m.add_class::<PyParsedResponse>()?;
    m.add_class::<PyParsedToolCall>()?;
    m.add_class::<PyToolCallParseStatus>()?;
    m.add_class::<PyQwen3VlImageProcessor>()?;
    Ok(())
}
