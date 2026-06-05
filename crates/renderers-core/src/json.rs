use std::io;

use serde::Serialize;
use serde_json::json;
use serde_json::ser::Formatter;

use crate::types::ToolSpec;

/// Serialize JSON with Python's default `json.dumps(..., ensure_ascii=False)`
/// separators: `", "` between values and `": "` between keys and values.
pub(crate) fn to_string_python<T>(value: &T) -> Result<String, serde_json::Error>
where
    T: Serialize + ?Sized,
{
    let mut out = Vec::new();
    {
        let mut serializer = serde_json::Serializer::with_formatter(&mut out, PythonJsonFormatter);
        value.serialize(&mut serializer)?;
    }
    Ok(String::from_utf8(out).expect("serde_json only writes valid UTF-8"))
}

pub(crate) fn tool_spec_inner_value(tool: &ToolSpec) -> serde_json::Value {
    json!({
        "name": &tool.name,
        "description": &tool.description,
        "parameters": &tool.parameters,
    })
}

pub(crate) fn tool_spec_openai_value(tool: &ToolSpec) -> serde_json::Value {
    json!({
        "type": "function",
        "function": tool_spec_inner_value(tool),
    })
}

pub(crate) fn tool_spec_template_value(tool: &ToolSpec) -> serde_json::Value {
    if tool.openai_envelope {
        tool_spec_openai_value(tool)
    } else {
        tool_spec_inner_value(tool)
    }
}

#[derive(Debug, Default)]
struct PythonJsonFormatter;

impl Formatter for PythonJsonFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }
}
