//! RAG retriever — combines index and store for top-k retrieval.

use super::index::RagIndex;
use super::store::{Snippet, SnippetStore};

/// Retrieves relevant snippets by searching the index and fetching from the store.
#[derive(Debug)]
pub struct Retriever<I: RagIndex, S: SnippetStore> {
    index: I,
    store: S,
}

impl<I: RagIndex, S: SnippetStore> Retriever<I, S> {
    /// Create a new retriever with the given index and store.
    pub fn new(index: I, store: S) -> Self {
        Self { index, store }
    }

    /// Add a snippet to both the index and the store.
    pub fn add(&mut self, snippet: Snippet) {
        self.index.index(&snippet.id, &snippet.content);
        self.store.put(snippet);
    }

    /// Search for relevant snippets given a query.
    ///
    /// Returns up to `top_k` snippets sorted by relevance.
    pub fn retrieve(&self, query: &str, top_k: usize) -> Vec<RetrievedSnippet> {
        let search_results = self.index.search(query, top_k);

        search_results
            .into_iter()
            .filter_map(|result| {
                self.store.get(&result.id).map(|snippet| RetrievedSnippet {
                    snippet: snippet.clone(),
                    score: result.score,
                })
            })
            .collect()
    }

    /// Remove a snippet from both the index and store.
    pub fn remove(&mut self, id: &str) {
        self.index.remove(id);
        self.store.remove(id);
    }

    /// Number of snippets stored.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Whether the retriever is empty.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Access the underlying index.
    pub fn index(&self) -> &I {
        &self.index
    }

    /// Access the underlying store.
    pub fn store(&self) -> &S {
        &self.store
    }
}

/// A snippet along with its relevance score from a search.
#[derive(Debug, Clone)]
pub struct RetrievedSnippet {
    /// The snippet content
    pub snippet: Snippet,
    /// Relevance score (higher = more relevant)
    pub score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::{InMemoryIndex, InMemoryStore, Snippet};

    #[test]
    fn test_retriever_add_and_search() {
        let index = InMemoryIndex::default();
        let store = InMemoryStore::default();
        let mut retriever = Retriever::new(index, store);

        retriever.add(Snippet {
            id: "1".to_string(),
            content: "def fibonacci(n): return fib(n-1) + fib(n-2)".to_string(),
            source: Some("math.py".to_string()),
            language: Some("python".to_string()),
        });

        retriever.add(Snippet {
            id: "2".to_string(),
            content: "fn main() { println!(\"hello\"); }".to_string(),
            source: Some("main.rs".to_string()),
            language: Some("rust".to_string()),
        });

        let results = retriever.retrieve("fibonacci", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].snippet.id, "1");
    }

    #[test]
    fn test_retriever_remove() {
        let index = InMemoryIndex::default();
        let store = InMemoryStore::default();
        let mut retriever = Retriever::new(index, store);

        retriever.add(Snippet {
            id: "1".to_string(),
            content: "test content".to_string(),
            source: None,
            language: None,
        });

        assert_eq!(retriever.len(), 1);
        retriever.remove("1");
        assert!(retriever.is_empty());
    }
}
