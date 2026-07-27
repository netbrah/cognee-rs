use std::sync::Arc;

use async_trait::async_trait;
use cognee_llm::GenerationOptions;
use cognee_session::SessionContext;

use crate::types::{SearchContext, SearchError, SearchOutput, SearchParams, SearchType};

pub type SearchRetrieverRef = Arc<dyn SearchRetriever>;

/// Minimum output-token cap for *internal machine generations* — LLM calls
/// whose output must parse, not be read by a human: NL→Cypher query synthesis
/// and the feeling-lucky retriever selector. This is the historical
/// [`GenerationOptions::default`] cap (16384); before it was applied here, a low
/// user-facing `LLM_MAX_COMPLETION_TOKENS` would flow into these calls and
/// truncate a Cypher query or a selector token into something unparseable.
pub(crate) const INTERNAL_GENERATION_MIN_TOKENS: u32 = 16_384;

/// Derive generation options for an internal machine generation from a
/// retriever's configured `generation_options`, guaranteeing the output-token
/// cap is at least [`INTERNAL_GENERATION_MIN_TOKENS`].
///
/// User-facing *answer* generation must keep honouring the configured cap
/// (that is what `LLM_MAX_COMPLETION_TOKENS` is for), so this helper is applied
/// only to the internal calls, never to the final-answer `generate` call.
pub(crate) fn floored_internal_options(
    base: &Option<GenerationOptions>,
) -> Option<GenerationOptions> {
    let mut opts = base.clone().unwrap_or_default();
    opts.max_tokens = Some(opts.max_tokens.map_or(INTERNAL_GENERATION_MIN_TOKENS, |n| {
        n.max(INTERNAL_GENERATION_MIN_TOKENS)
    }));
    Some(opts)
}

#[async_trait]
pub trait SearchRetriever: Send + Sync {
    fn search_type(&self) -> SearchType;

    async fn get_context(
        &self,
        query: &str,
        params: &SearchParams,
    ) -> Result<SearchContext, SearchError>;

    async fn get_completion(
        &self,
        query: &str,
        context: Option<SearchContext>,
        session: &SessionContext,
        params: &SearchParams,
    ) -> Result<SearchOutput, SearchError>;

    /// Process multiple queries in sequence and return their contexts.
    ///
    /// Default: loops over [`get_context`]. Override for efficient batching.
    async fn get_context_batch(
        &self,
        queries: &[String],
        params: &SearchParams,
    ) -> Result<Vec<SearchContext>, SearchError> {
        let mut results = Vec::with_capacity(queries.len());
        for query in queries {
            results.push(self.get_context(query, params).await?);
        }
        Ok(results)
    }

    /// Process multiple queries and return their completions.
    ///
    /// Default: loops over [`get_completion`]. Override for efficient batching.
    async fn get_completion_batch(
        &self,
        queries: &[String],
        contexts: Option<Vec<SearchContext>>,
        session: &SessionContext,
        params: &SearchParams,
    ) -> Result<Vec<SearchOutput>, SearchError> {
        let mut results = Vec::with_capacity(queries.len());
        for (i, query) in queries.iter().enumerate() {
            let ctx = contexts.as_ref().and_then(|cs| cs.get(i).cloned());
            results.push(self.get_completion(query, ctx, session, params).await?);
        }
        Ok(results)
    }
}
