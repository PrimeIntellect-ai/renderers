//! [`MediaResolver`] implementations backed by the in-crate vision
//! processors. Lets pure-Rust callers go from "image bytes / URL /
//! path" straight to a [`MediaItem`] without a Python round-trip.

use std::fs;

use serde_json::json;

use crate::processing::qwen3_vl::{ProcessedImage, Qwen3VlImageProcessor};
use crate::traits::{MediaResolver, MediaSource};
use crate::types::{MediaItem, Modality, RenderError};

/// `MediaResolver` backed by [`Qwen3VlImageProcessor`]. Stores the
/// processed tensor inside `MediaItem.hf_payload` as a JSON object so
/// the inference engine glue can route it through the same path as
/// the Python-resolved case.
///
/// The serialised payload shape is:
///
/// ```json
/// {
///   "pixel_values":   { "shape": [tokens, features], "data": [f32, ...] },
///   "image_grid_thw": { "shape": [1, 3],             "data": [1, h, w]  }
/// }
/// ```
///
/// Callers that need zero-copy `numpy`/`torch` arrays should consume
/// the [`ProcessedImage`] struct directly via
/// [`Qwen3VlResolver::process_bytes`] instead of going through the
/// `MediaItem.hf_payload` field.
#[derive(Debug, Clone, Default)]
pub struct Qwen3VlResolver {
    processor: Qwen3VlImageProcessor,
}

impl Qwen3VlResolver {
    pub fn new(processor: Qwen3VlImageProcessor) -> Self {
        Self { processor }
    }

    pub fn processor(&self) -> &Qwen3VlImageProcessor {
        &self.processor
    }

    /// Process raw image bytes into the structured [`ProcessedImage`]
    /// — the zero-loss representation. The [`MediaResolver`] impl
    /// wraps this and re-serialises into `MediaItem.hf_payload`.
    pub fn process_bytes(&self, bytes: &[u8]) -> Result<ProcessedImage, RenderError> {
        self.processor.process_bytes(bytes)
    }

    fn to_media_item(processed: ProcessedImage) -> MediaItem {
        let shape = processed.pixel_values.shape();
        let pixel_shape = vec![shape[0] as u64, shape[1] as u64];
        let pixel_data: Vec<f32> = processed.pixel_values.iter().copied().collect();
        let grid: Vec<u32> = processed.image_grid_thw.to_vec();

        let payload = json!({
            "pixel_values": {
                "shape": pixel_shape,
                "data":  pixel_data,
            },
            "image_grid_thw": {
                "shape": [1u32, 3u32],
                "data":  grid,
            },
        });

        MediaItem {
            modality: Modality::Image,
            hash: processed.hash,
            num_tokens: processed.num_tokens,
            hf_payload: payload,
        }
    }
}

impl MediaResolver for Qwen3VlResolver {
    fn resolve_image(&self, source: &MediaSource<'_>) -> Result<MediaItem, RenderError> {
        let bytes: Vec<u8> = match source {
            MediaSource::Bytes(b) => b.to_vec(),
            MediaSource::Path(p) => fs::read(p)
                .map_err(|e| RenderError::Invalid(format!("read image {}: {e}", p.display())))?,
            MediaSource::Url(_) => {
                return Err(RenderError::Invalid(
                    "URL sources require an async fetch — pass already-downloaded bytes instead"
                        .into(),
                ));
            }
        };
        let processed = self.process_bytes(&bytes)?;
        Ok(Self::to_media_item(processed))
    }
}
