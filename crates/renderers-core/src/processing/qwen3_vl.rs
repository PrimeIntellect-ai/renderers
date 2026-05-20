//! Vision image processing for Qwen-VL family models (Qwen2-VL,
//! Qwen3-VL, Qwen3.5-VL).
//!
//! Port of the HuggingFace `Qwen2VLImageProcessor` / `Qwen3VLImageProcessor`
//! pipeline. Given an image (bytes or decoded RGB), produces:
//!
//! - `pixel_values`: `ndarray::Array2<f32>` of shape
//!   `(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)`.
//!   This is what the vision encoder consumes.
//! - `image_grid_thw`: `[1, grid_h, grid_w]` — the temporal × height × width
//!   patch count.
//! - `num_tokens`: `grid_h * grid_w / (merge_size * merge_size)` — the
//!   placeholder count the renderer emits between
//!   `<|vision_start|>` and `<|vision_end|>`.
//!
//! # Parity caveat
//!
//! The grid dimensions, num_tokens, and tensor shape match HF exactly.
//! The pixel values themselves use the `image` crate's bicubic
//! (CatmullRom) resize, which differs from PIL's bicubic in the last
//! few decimals — typical RMS difference ≈ 1e-3 on normalized pixels.
//! Downstream models tolerate this level of noise (it's far below the
//! quantization floor of vision encoders); but if exact pixel parity
//! is required (e.g. for regression tests against PIL-rendered
//! fixtures) keep the Python processor on the path.

use std::io::Cursor;

use ndarray::{Array2, Array3};
use sha2::{Digest, Sha256};

use crate::types::RenderError;

/// OpenAI CLIP normalisation constants — Qwen-VL inherits these.
pub const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Configuration for the Qwen-VL image processor pipeline.
#[derive(Debug, Clone)]
pub struct Qwen3VlImageProcessor {
    /// Lower bound on resized pixel count. Default for Qwen2-VL / Qwen3-VL:
    /// `56 * 56 = 3136`. Resized images smaller than this get scaled up.
    pub min_pixels: u32,
    /// Upper bound on resized pixel count. Default: `28*28*1280 = 1_003_520`.
    pub max_pixels: u32,
    /// Patch size in pixels. Default: 14.
    pub patch_size: u32,
    /// Temporal patch size — `pixel_values` is duplicated across this
    /// axis for static images so the same tensor shape serves images
    /// and video frames. Default: 2.
    pub temporal_patch_size: u32,
    /// Spatial merge factor between vision encoder output and the
    /// model's input — placeholders count divides by `merge²`. Default: 2.
    pub merge_size: u32,
    /// Rescale factor applied before normalisation. Default: 1/255.
    pub rescale_factor: f32,
    /// Per-channel mean / std for normalisation (after rescale).
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Default for Qwen3VlImageProcessor {
    fn default() -> Self {
        Self {
            min_pixels: 56 * 56,
            max_pixels: 28 * 28 * 1280,
            patch_size: 14,
            temporal_patch_size: 2,
            merge_size: 2,
            rescale_factor: 1.0 / 255.0,
            image_mean: CLIP_MEAN,
            image_std: CLIP_STD,
        }
    }
}

/// Output of one image's processing run.
#[derive(Debug, Clone)]
pub struct ProcessedImage {
    /// Flattened patches: shape (grid_h * grid_w, channel * temporal * patch²).
    pub pixel_values: Array2<f32>,
    /// `[1, grid_h, grid_w]` — temporal × height × width patch count.
    pub image_grid_thw: [u32; 3],
    /// `grid_h * grid_w / merge²` — count of placeholder tokens to emit.
    pub num_tokens: usize,
    /// Stable SHA-256 prefix of the resolved RGB bytes — useful as a
    /// cache key.
    pub hash: String,
}

impl Qwen3VlImageProcessor {
    /// Compute the resized (height, width) for an input image. Mirrors
    /// `transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize`.
    ///
    /// `factor = patch_size * merge_size` (28 by default).
    pub fn smart_resize(&self, height: u32, width: u32) -> Result<(u32, u32), RenderError> {
        let factor = self.patch_size * self.merge_size;
        let (h, w) = (height as f64, width as f64);
        let max_dim = h.max(w);
        let min_dim = h.min(w);
        if min_dim == 0.0 {
            return Err(RenderError::Invalid("image dimension is zero".into()));
        }
        if max_dim / min_dim > 200.0 {
            return Err(RenderError::Invalid(format!(
                "absolute aspect ratio must be smaller than 200, got {:.2}",
                max_dim / min_dim
            )));
        }
        let f = factor as f64;
        let mut h_bar = (h / f).round() * f;
        let mut w_bar = (w / f).round() * f;

        let max_pixels = self.max_pixels as f64;
        let min_pixels = self.min_pixels as f64;

        if h_bar * w_bar > max_pixels {
            let beta = ((h * w) / max_pixels).sqrt();
            h_bar = ((h / beta) / f).floor() * f;
            w_bar = ((w / beta) / f).floor() * f;
            h_bar = h_bar.max(f);
            w_bar = w_bar.max(f);
        } else if h_bar * w_bar < min_pixels {
            let beta = (min_pixels / (h * w)).sqrt();
            h_bar = ((h * beta) / f).ceil() * f;
            w_bar = ((w * beta) / f).ceil() * f;
        }
        Ok((h_bar as u32, w_bar as u32))
    }

    /// Decode arbitrary image bytes (PNG/JPEG/WebP via the `image`
    /// crate's auto-detect) to RGB pixel arrays.
    pub fn decode(bytes: &[u8]) -> Result<image::RgbImage, RenderError> {
        let reader = image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|e| RenderError::Invalid(format!("image format detection: {e}")))?;
        let dynamic = reader
            .decode()
            .map_err(|e| RenderError::Invalid(format!("image decode: {e}")))?;
        Ok(dynamic.to_rgb8())
    }

    /// Hash the resolved RGB bytes — same shape as the Python
    /// `_image_hash` so the cache key is comparable.
    pub fn hash_rgb(rgb: &image::RgbImage) -> String {
        let mut h = Sha256::new();
        h.update(rgb.as_raw());
        h.update(format!("({}, {})", rgb.width(), rgb.height()).as_bytes());
        let digest = h.finalize();
        // Trim to 32 hex chars to match the Python implementation.
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        hex[..32].to_string()
    }

    /// Process a single decoded RGB image end-to-end.
    pub fn process_rgb(&self, rgb: &image::RgbImage) -> Result<ProcessedImage, RenderError> {
        let (orig_w, orig_h) = (rgb.width(), rgb.height());
        let (new_h, new_w) = self.smart_resize(orig_h, orig_w)?;

        // Resize: image crate's CatmullRom is the closest match to PIL's
        // bicubic. See module-level docs for the parity caveat.
        let resized =
            image::imageops::resize(rgb, new_w, new_h, image::imageops::FilterType::CatmullRom);

        // Build a (C=3, H, W) f32 array, normalised.
        let (h, w) = (new_h as usize, new_w as usize);
        let mut chw = Array3::<f32>::zeros((3, h, w));
        for y in 0..h {
            for x in 0..w {
                let p = resized.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let v = (p[c] as f32) * self.rescale_factor;
                    chw[(c, y, x)] = (v - self.image_mean[c]) / self.image_std[c];
                }
            }
        }

        // Patch layout. The HF pipeline reshapes to:
        //   (C, grid_h/merge, merge, patch, grid_w/merge, merge, patch)
        // then permutes to:
        //   (grid_h/merge, grid_w/merge, merge, merge, C, patch, patch)
        // then unsqueezes a temporal axis and expands to temporal_patch_size,
        // finally flattening to (grid_h*grid_w, C*temporal*patch*patch).
        //
        // The output layout is (token_idx, feature) where token_idx
        // iterates in row-major order over the merged grid:
        //   token_idx = (m_row * grid_w/merge + m_col) * merge² + mi*merge + mj
        // and the feature vector packs (C, temporal, patch, patch) in
        // row-major order.
        let ps = self.patch_size as usize;
        let merge = self.merge_size as usize;
        let temporal = self.temporal_patch_size as usize;
        let grid_h = h / ps;
        let grid_w = w / ps;
        if grid_h % merge != 0 || grid_w % merge != 0 {
            return Err(RenderError::Invalid(format!(
                "resized grid ({grid_h}x{grid_w}) not divisible by merge_size {merge}"
            )));
        }
        let token_count = grid_h * grid_w;
        let feature_len = 3 * temporal * ps * ps;
        let mut pixel_values = Array2::<f32>::zeros((token_count, feature_len));

        // Fill: for each token (m_row, m_col, mi, mj), copy the corresponding
        // (patch_size × patch_size × 3) sub-block, replicated across the
        // temporal axis.
        let merged_grid_h = grid_h / merge;
        let merged_grid_w = grid_w / merge;
        for m_row in 0..merged_grid_h {
            for m_col in 0..merged_grid_w {
                for mi in 0..merge {
                    for mj in 0..merge {
                        let token_idx = ((m_row * merged_grid_w + m_col) * merge + mi) * merge + mj;
                        // Patch top-left in pixel coordinates:
                        let py = (m_row * merge + mi) * ps;
                        let px = (m_col * merge + mj) * ps;
                        let mut feature_idx = 0usize;
                        for c in 0..3 {
                            for _t in 0..temporal {
                                for dy in 0..ps {
                                    for dx in 0..ps {
                                        pixel_values[(token_idx, feature_idx)] =
                                            chw[(c, py + dy, px + dx)];
                                        feature_idx += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let num_tokens = (grid_h * grid_w) / (merge * merge);
        let hash = Self::hash_rgb(rgb);

        Ok(ProcessedImage {
            pixel_values,
            image_grid_thw: [1, grid_h as u32, grid_w as u32],
            num_tokens,
            hash,
        })
    }

    /// Convenience: decode bytes then process.
    pub fn process_bytes(&self, bytes: &[u8]) -> Result<ProcessedImage, RenderError> {
        let rgb = Self::decode(bytes)?;
        self.process_rgb(&rgb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_round_trip() {
        let p = Qwen3VlImageProcessor::default();
        let (h, w) = p.smart_resize(480, 640).unwrap();
        // 480*640 = 307_200 → under max_pixels, both align to factor 28
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn smart_resize_scales_down_oversized() {
        let p = Qwen3VlImageProcessor::default();
        // 4000*3000 = 12M pixels — must scale down
        let (h, w) = p.smart_resize(4000, 3000).unwrap();
        assert!(h * w <= p.max_pixels);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn smart_resize_scales_up_undersized() {
        let p = Qwen3VlImageProcessor::default();
        // 16x16 = 256 pixels — below min, must scale up
        let (h, w) = p.smart_resize(16, 16).unwrap();
        assert!(h * w >= p.min_pixels);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn smart_resize_rejects_extreme_aspect_ratio() {
        let p = Qwen3VlImageProcessor::default();
        assert!(p.smart_resize(10, 10_000).is_err());
    }

    #[test]
    fn process_small_image() {
        let p = Qwen3VlImageProcessor::default();
        // Synthesise a 56x56 RGB image
        let mut rgb = image::RgbImage::new(56, 56);
        for y in 0..56 {
            for x in 0..56 {
                rgb.put_pixel(x, y, image::Rgb([x as u8, y as u8, 128]));
            }
        }
        let out = p.process_rgb(&rgb).unwrap();
        assert_eq!(out.image_grid_thw, [1, 4, 4]);
        assert_eq!(out.num_tokens, 4); // 16 / (2*2)
        // pixel_values shape: (16 tokens, 3*2*14*14 = 1176)
        assert_eq!(out.pixel_values.shape(), &[16, 1176]);
    }
}
