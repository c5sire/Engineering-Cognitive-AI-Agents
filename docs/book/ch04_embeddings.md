# Design Document: `EmbeddingStore`

## Overview

The `EmbeddingStore` class provides efficient storage, retrieval, and management of knowledge embeddings using **ChromaDB**, a high-performance vector database. By leveraging embeddings, this system supports semantic searches and similarity matching to identify knowledge entries based on their contextual meaning.

This implementation is built upon the `Knowledge` model from `KnowledgeStorage`, providing tight integration between structured knowledge storage and embedding-based search capabilities.

---

## Goals

- **Efficient Vector Storage**: Use ChromaDB to manage knowledge embeddings in a performant and scalable way.
- **Similarity Search**: Enable queries to find semantically similar knowledge entries.
- **CRUD for Embeddings**:
  - Add, update, and delete embeddings in the backing ChromaDB store.
- **Integration with Metadata**: Support filtering search results based on metadata context.
- **Scalable and Persistent**: Ensure the system can support large-scale embeddings and persistent storage.

---

## Scope and Features

### Current Features

1. **Embedding Storage**:

   - Embeddings are stored in **ChromaDB** as vector representations of knowledge content.
   - Uses ChromaDB's `PersistentClient` for long-term storage on disk.

2. **Similarity Search**:

   - Finds knowledge entries semantically similar to a query using **cosine similarity**.
     - Adjusted scores (`1.0 - distance`) are returned for easier interpretation.
   - Supports metadata-based filtering during search.

3. **CRUD Operations**:

   - **Add Embeddings**: Converts `Knowledge` entries to embeddings and stores them in ChromaDB.
   - **Update Embeddings**: Updates embeddings of modified `Knowledge` entries.
   - **Delete Embeddings**: Removes embeddings from ChromaDB based on knowledge IDs.

4. **Metadata Integration**:

   - Embeddings include metadata from the `Knowledge` model.
   - Search results return the associated metadata for contextual visibility.

5. **Scalability**:
   - Designed to handle large-scale knowledge embeddings using ChromaDB.
   - Disk-based storage ensures persistence and efficient retrieval.

---

## Design Details

### Architecture

1. **ChromaDB PersistentClient**:

   - The `EmbeddingStore` uses a `PersistentClient` from ChromaDB for embedding storage.
   - Storage path is provided during initialization.
   - Embeddings are organized into a single collection: `"knowledge_embeddings"`.

2. **Embedding Representation**:

   - Knowledge content (`Knowledge.content`) is embedded as dense vectors (generated internally by ChromaDB).
   - Metadata from `Knowledge.context` is associated with vectors to enable contextual filtering.

3. **Similarity Search**:

   - Uses ChromaDB's native querying capabilities with **cosine similarity**.
   - Results include calculated similarity scores (`1 - distance`) for intuitive ranking.

4. **CRUD Implementation**:

   - **Add**: Embedding vectors and IDs are stored in the ChromaDB collection.
   - **Update**: Replaces embedding vectors, metadata, and IDs in the collection.
   - **Delete**: Removes vectors by ID from the collection via ChromaDB's `delete` method.

5. **Integration with Knowledge Model**:
   - Tight coupling with the `Knowledge` model ensures consistent behavior across storage and embeddings.
   - Metadata (`context`) is passed directly from the `Knowledge` object.

---

### Operations

#### `add_embedding`

- **Purpose**: Add a new knowledge entry embedding to the ChromaDB store.
- **Input**: A `Knowledge` object.
- **Behavior**:
  - Converts `Knowledge.content` into an embedding.
  - Saves the embedding along with its metadata (`Knowledge.context`) and ID (`Knowledge.id`) in the ChromaDB collection.

---

#### `find_similar`

- **Purpose**: Query the ChromaDB store for knowledge entries semantically close to a given text query.
- **Input**:
  - `query`: Query text to match against embeddings.
  - `limit`: Maximum number of results to return (default: 5).
  - `filters`: Metadata filters (optional) to further narrow results.
- **Output**: A list of `SimilarityMatch` objects containing:
  - `id`: Knowledge ID of the matching entry.
  - `score`: Similarity score (higher value indicates closer match).
  - `metadata`: Contextual metadata of the matching entry.
- **Behavior**:
  - Combines query text with optional metadata filters.
  - Executes a similarity search using cosine distance.
  - Converts distances to similarity scores (`1.0 - distance`).

---

#### `update_embedding`

- **Purpose**: Update the stored embedding for an existing knowledge entry (e.g., after content changes).
- **Input**: Updated `Knowledge` object.
- **Behavior**:
  - Finds corresponding embedding by `Knowledge.id`.
  - Updates the embedding content and metadata in the ChromaDB collection.

---

#### `delete_embedding`

- **Purpose**: Remove an existing embedding from the store.
- **Input**: Knowledge ID to delete.
- **Behavior**:
  - Deletes the embedding by ID from the ChromaDB collection.

---

### Extensibility

- **Filters**: Uses ChromaDB's built-in metadata filtering to support narrowing results based on key-value filters.
- **Custom Embeddings**: Although ChromaDB's default embedding functionality is used here, external embeddings could be added by replacing the default.

---

### Data Flow

1. **Storage**:
   - Knowledge entry content (`Knowledge.content`) → Vector embedding.
   - Knowledge entry metadata (`Knowledge.context`) → Metadata key-value pairs.
2. **Search**:
   - Query → Vector representation → Cosine similarity comparison with the embedded vectors.
   - Results include IDs, metadata, and similarity scores.

---

## Code Standards

- **Pydantic Validation**: Validates `Knowledge` input fields and ensures metadata consistency.
- **Type Hints**: Ensures clear understanding of input/output types.
- **ChromaDB Integration**: Wrapper methods simplify calls to ChromaDB operations (add, query, update, delete).
- **Use of NamedTuple**: `SimilarityMatch` is designed for lightweight result representation.

---

## Test Plan

### Unit Tests

Tested features include:

1. **Add Embeddings**:

   - Ensure vectors are stored correctly in ChromaDB.
   - Verify consistency of metadata storage.

2. **Similarity Search**:

   - Query the embeddings for semantic matches.
   - Check results for:
     - Correct ranking based on similarity.
     - Returned metadata and IDs match the stored entries.
   - Test with and without metadata filters.

3. **Update Embedding**:

   - Update an existing embedding and ensure the changes reflect in future searches.

4. **Delete Embedding**:

   - Remove an embedding and confirm it can no longer be queried.

5. **Edge Cases**:
   - Search with no stored embeddings (expect empty results).
   - Search with irrelevant query text (expect no matches).
   - Delete nonexistent embeddings (ensure no crash).

---

### Example Test Cases (from `test_embeddings.py`)

- **Add Embeddings and Search**:
  - Add two sample `Knowledge` entries and verify similarity results.
  - Test both query text and filtered searches.
- **Update Embedding**:
  - Modify an entry's content and make sure future search results reflect the updated embedding.
- **Delete Embedding**:
  - Ensure a deleted entry no longer appears in search queries.

---

## Tradeoffs and Future Considerations

### Tradeoffs

1. **Custom Embeddings**:

   - The current implementation relies on ChromaDB's built-in embedding generation, which might limit flexibility. External embedding models (e.g., OpenAI or Hugging Face) could enhance semantic accuracy but introduce complexity.

2. **Complex Metadata**:

   - The system supports only flat key-value metadata, which might not handle large or deeply nested metadata efficiently.

3. **File-based Persistence**:
   - While simple and effective for small-to-medium datasets, scaling to large datasets might require migrating to distributed ChromaDB deployments.

---

### Future Features

1. **Custom Embedding Models**:

   - Integrate external embedding generation for domain-specific applications.

2. **Improved Metadata Handling**:

   - Support nested or hierarchical metadata.

3. **Ranking Tuning**:

   - Allow users to customize distance conversion metrics or incorporate other ranking heuristics.

4. **Versioning**:

   - Track changes to embeddings over time to allow result audits.

5. **Sharded Databases**:
   - Support for distributed storage for scalability as stored knowledge grows.

---

## Summary

The `EmbeddingStore` implementation provides an efficient, scalable, and flexible system for managing and querying embeddings using ChromaDB. It enables semantic similarity searches, integrates with the `Knowledge` model, and delivers core CRUD operations for embeddings. Future enhancements will focus on increasing scalability and embedding flexibility.
