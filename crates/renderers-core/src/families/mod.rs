//! Per-family renderer implementations.
//!
//! Each family lives in its own module so the hand-coded template logic
//! stays focused. New families slot in by adding a module here and a
//! registry entry in [`crate::registry`].

pub mod deepseek_v3;
pub mod qwen3;
pub mod qwen35;
pub mod qwen36;

pub use deepseek_v3::{DeepSeekV3Renderer, DeepSeekV3RendererBuilder};
pub use qwen3::{Qwen3Renderer, Qwen3RendererBuilder};
pub use qwen35::{Qwen35Renderer, Qwen35RendererBuilder};
pub use qwen36::{Qwen36Renderer, Qwen36RendererBuilder};
