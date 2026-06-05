use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use crate::tokenizer::Tokenizer;
use crate::types::{RenderError, ToolSpec};

const MAX_TOOL_TEXT_CACHE_ENTRIES: usize = 64;

#[derive(Debug, Clone, Default)]
pub(crate) struct ToolTextCache {
    inner: Arc<Mutex<HashMap<ToolTextCacheKey, CachedToolText>>>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct ToolTextCacheKey {
    tools_ptr: usize,
    tools_len: usize,
    discriminator: u64,
    dynamic_hash: u64,
}

#[derive(Debug, Clone)]
struct CachedToolText {
    tools: Vec<ToolSpec>,
    dynamic_text: String,
    tokens: Arc<Vec<u32>>,
}

impl ToolTextCache {
    pub(crate) fn get_or_insert_with(
        &self,
        tokenizer: &Tokenizer,
        tools: &[ToolSpec],
        discriminator: u64,
        dynamic_text: &str,
        build_text: impl FnOnce() -> Result<String, RenderError>,
    ) -> Result<Arc<Vec<u32>>, RenderError> {
        let key = ToolTextCacheKey {
            tools_ptr: tools.as_ptr() as usize,
            tools_len: tools.len(),
            discriminator,
            dynamic_hash: hash_dynamic_text(dynamic_text),
        };

        {
            let cache = self.lock_cache()?;
            if let Some(cached) = cache.get(&key) {
                if cached.tools == tools && cached.dynamic_text == dynamic_text {
                    return Ok(cached.tokens.clone());
                }
            }
        }

        let text = build_text()?;
        let tokens = Arc::new(tokenizer.encode_no_special(&text)?.as_slice().to_vec());
        let mut cache = self.lock_cache()?;
        if cache.len() >= MAX_TOOL_TEXT_CACHE_ENTRIES {
            cache.clear();
        }
        cache.insert(
            key,
            CachedToolText {
                tools: tools.to_vec(),
                dynamic_text: dynamic_text.to_string(),
                tokens: tokens.clone(),
            },
        );
        Ok(tokens)
    }

    fn lock_cache(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, HashMap<ToolTextCacheKey, CachedToolText>>, RenderError>
    {
        self.inner
            .lock()
            .map_err(|_| RenderError::Invalid("tool text cache lock poisoned".into()))
    }
}

fn hash_dynamic_text(text: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}
