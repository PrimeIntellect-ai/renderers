//! Vision processors — port of the HuggingFace image processor pipelines.
//!
//! Phase 5b: actual pixel-data preprocessing in Rust. Decode image bytes,
//! smart-resize, normalise, patch-extract, and produce the tensors the
//! vision encoder consumes — same shape as HF's processors, without
//! crossing back to Python.
//!
//! Currently shipped:
//!
//! - [`qwen3_vl::Qwen3VlImageProcessor`] — covers Qwen2-VL, Qwen3-VL,
//!   and Qwen3.5-VL (they share the processor).
//!
//! Future:
//!
//! - Kimi K2.5 — different smart_resize defaults and a single-pad
//!   placeholder convention (Phase 5b follow-up).
//! - Video frame sampling — needs `video-rs` or `ffmpeg-next` (Phase 5c).

pub mod qwen3_vl;
pub mod resolver;

pub use qwen3_vl::{CLIP_MEAN, CLIP_STD, ProcessedImage, Qwen3VlImageProcessor};
pub use resolver::Qwen3VlResolver;
