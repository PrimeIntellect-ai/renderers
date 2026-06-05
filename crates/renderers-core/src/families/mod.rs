//! Per-family renderer implementations.
//!
//! Each family lives in its own module so the hand-coded template logic
//! stays focused. New families slot in by adding a module here and a
//! registry entry in [`crate::registry`].

pub mod deepseek_v3;
pub mod default;
pub mod glm;
pub mod gpt_oss;
pub mod kimi_k2;
pub mod kimi_k25;
pub mod minimax_m2;
pub mod nemotron3;
pub mod qwen3;
pub mod qwen35;
pub mod qwen36;

pub use deepseek_v3::{DeepSeekV3Renderer, DeepSeekV3RendererBuilder};
pub use default::{DefaultRenderer, DefaultRendererBuilder};
pub use glm::{GlmRenderer, GlmRendererBuilder};
pub use gpt_oss::{GptOssRenderer, GptOssRendererBuilder};
pub use kimi_k2::{KimiK2Renderer, KimiK2RendererBuilder};
pub use kimi_k25::{KimiK25Renderer, KimiK25RendererBuilder};
pub use minimax_m2::{MiniMaxM2Renderer, MiniMaxM2RendererBuilder};
pub use nemotron3::{Nemotron3Renderer, Nemotron3RendererBuilder};
pub use qwen3::{Qwen3Renderer, Qwen3RendererBuilder};
pub use qwen35::{Qwen35Renderer, Qwen35RendererBuilder};
pub use qwen36::{Qwen36Renderer, Qwen36RendererBuilder};
