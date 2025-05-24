# Design Document: ChromaDB Integration for DiscoveryAgent

**Date**: <Current Date>
**Author**: BeatDebate Team (AI Assisted)
**Status**: Revised Draft
**Review Status**: Pending

---

## 1. Problem Statement

The `DiscoveryAgent` currently relies primarily on the Last.fm API for finding similar artists and tracks. Integrating ChromaDB aims to:
*   Establish a local vector store for future custom similarity metrics.
*   Reduce external API calls in the long term.
*   Build a foundation for advanced, nuanced discovery features.

For the MVP, this integration focuses on **setting up ChromaDB within the DiscoveryAgent and populating it with basic track data and default embeddings**. The primary similarity mechanism for generating recommendations in the `DiscoveryAgent` during the MVP phase may still rely on Last.fm's existing similarity functions. Querying ChromaDB in the MVP will be considered experimental or a secondary source.

The long-term goal (post-MVP) is to transition the `DiscoveryAgent` to use ChromaDB with custom-generated embeddings as its primary source for similarity searches.

---

## 2. Goals & Non-Goals

### ✅ In Scope (MVP Integration)
- Initialize and manage a ChromaDB client (persistent by default) and a dedicated collection for music tracks within the `DiscoveryAgent`.
- As tracks are processed by the `DiscoveryAgent` (e.g., from Last.fm), populate ChromaDB by:
    - Storing basic track metadata (title, artist, Last.fm URL, listeners).
    - Generating embeddings for simple text representations (e.g., "Artist - Title") using ChromaDB's default embedding model.
- Basic error handling for ChromaDB population operations.
- Unit tests for ChromaDB setup and population logic.
- Optional/experimental: `DiscoveryAgent` may perform basic queries against ChromaDB, but these results are not critical for the primary MVP recommendation output.

### ❌ Out of Scope (for this MVP integration)
- ChromaDB serving as the *primary* source for similarity-based recommendations in the `DiscoveryAgent`. (Primary reliance will be on Last.fm's similarity for the MVP).
- Implementation of *custom, rich embedding strategies* (e.g., combining genre, tags, mood as per the main `beatdebate-design-doc.md`). This is a **future task**.
- Building a UI for managing or visualizing ChromaDB.
- Advanced embedding model selection, fine-tuning, or comparison.
- Complex data synchronization strategies between Last.fm and ChromaDB.

---

## 3. Proposed Solution (MVP Focus)

### 3.1 ChromaDB Setup & Initialization

-   **Client**: The `DiscoveryAgent` will initialize a `chromadb.PersistentClient()` upon its instantiation.
    -   The path for persistence (e.g., `"./chroma_data"`) will be configurable, defaulting to a local directory.
    ```python
    # Example in DiscoveryAgent.__init__
    # self.chroma_client = chromadb.PersistentClient(path=self.config.chromadb_path or "./chroma_data")
    ```
-   **Collection**: A dedicated collection, `"music_tracks"`, will be created or retrieved.
    ```python
    # Example in DiscoveryAgent.__init__
    # self.track_collection = self.chroma_client.get_or_create_collection(name="music_tracks")
    ```
-   ChromaDB's default embedding function (e.g., Sentence Transformers `all-MiniLM-L6-v2`) will be used.

### 3.2 Data Model in ChromaDB (for MVP Population)

-   **Documents**: For MVP population, the text data to be embedded will be simple: e.g., `f"{artist_name} - {track_title}"`.
-   **Embeddings**: Generated automatically by ChromaDB using its default model.
-   **Metadatas**:
    ```python
    {
        "artist": "Artist Name",
        "title": "Track Title",
        "lastfm_url": "http://last.fm/...",
        "listeners": 12345, // Integer or string
        "source": "lastfm" // Identifies the origin of the data
    }
    ```
-   **IDs**: Unique string identifiers, e.g., `f"lastfm_{normalized_artist}_{normalized_track}"`.

### 3.3 Embedding Strategy (for MVP Population)

-   **Embedding Model**: Use ChromaDB's default Sentence Transformer model.
-   **Data to Embed (MVP)**: A concise string: `f"{track_artist} - {track_name}"`. This is for initial population; this document string will be enriched in future phases when custom embeddings are implemented.
-   **Population Strategy**:
    -   **Opportunistic Addition**: When the `DiscoveryAgent` processes track details (e.g., from Last.fm calls made for its primary recommendation logic), this data will be formatted and added to the `music_tracks` collection in ChromaDB.

### 3.4 Querying ChromaDB (Experimental in MVP)

-   While the `DiscoveryAgent` *can* be equipped to query ChromaDB, for the MVP, this functionality is secondary or experimental.
    -   `DiscoveryAgent`'s primary recommendations will likely stem from Last.fm's similarity logic.
    -   Queries to ChromaDB, if implemented, would use text like `"{artist_name} - {track_title}"` of a seed track.
-   The focus is on robustly populating ChromaDB rather than perfecting its query performance for recommendations in this phase.

### 3.5 Integration with Existing Logic (MVP Hybrid Approach)

-   **Primary Logic**: `DiscoveryAgent` uses Last.fm API for its core similarity assessments and to identify candidate tracks for recommendations.
-   **Parallel Population**: Tracks retrieved from Last.fm are then processed and added to ChromaDB (embedding the simple `"{artist} - {title}"` string).
-   This ensures ChromaDB is populated without making it a critical path for MVP recommendations.

---

## 4. Key Changes to `DiscoveryAgent`

-   **`__init__(self, config: AgentConfig, ...)`**:
    -   `AgentConfig` (or similar) should provide `chromadb_path`.
    -   Initialize `self.chroma_client` as `chromadb.PersistentClient`.
    -   Initialize `self.track_collection`.
-   **New Method: `_add_tracks_to_chromadb(self, tracks: List[Dict[str, Any]])`**:
    -   Transforms track dictionaries into the simple ChromaDB data model for MVP (embedding `"{artist} - {title}"`).
    -   Handles batch adding using `self.track_collection.add(...)`.
-   **Modified Method: `_explore_similar_music(...)` (or relevant method)**:
    -   Continues to use Last.fm for primary similarity and candidate generation for MVP.
    -   After fetching from Last.fm, calls `_add_tracks_to_chromadb` to populate the local store.
    -   May include an optional, experimental step to query ChromaDB, but these results are not critical for MVP output.
-   **Helper methods**:
    -   `_format_track_for_chroma_mvp(track_data: Dict) -> Tuple[str, Dict, str]`: Converts track to MVP Chroma document, metadata, ID.

---

## 5. Data Model for ChromaDB Collection ("music_tracks") (MVP Population)

-   **Collection Name**: `music_tracks`
-   **Documents (MVP)**: Text string: `"{artist} - {title}"`. (Note: To be enriched in future phases for custom embedding strategy).
-   **Metadatas**:
    -   `artist: str`
    -   `title: str`
    -   `lastfm_url: Optional[str]`
    -   `listeners: Optional[Any]` (store as int if possible, or string)
    -   `source: str` (e.g., "lastfm")
    -   `added_timestamp: float`
-   **IDs**: `str`, e.g., `f"lastfm_{normalized_artist}_{normalized_title}"`

---

## 6. Implementation Plan (MVP)

1.  **Setup ChromaDB**: Add `chromadb` to project dependencies.
2.  **Update `DiscoveryAgent.__init__`**: Implement persistent ChromaDB client and collection initialization using configurable path (from `AgentConfig` or similar).
3.  **Implement `_format_track_for_chroma_mvp` and `_add_tracks_to_chromadb`**: Focus on simple text embedding and batch adding.
4.  **Integrate into `_explore_similar_music` (or equivalent)**: Ensure tracks from Last.fm are passed to `_add_tracks_to_chromadb`.
5.  **(Optional/Experimental for MVP)** Implement basic `_query_chroma_for_similar` if desired, but not as a primary recommendation source.
6.  **Testing**: Unit tests for ChromaDB setup, formatting, and adding logic.

---

## 7. Testing Strategy (MVP)

-   **Unit Tests**:
    -   Test `DiscoveryAgent` initialization with ChromaDB (persistent client).
    -   Test `_format_track_for_chroma_mvp`.
    -   Test `_add_tracks_to_chromadb` (mock collection `add`, verify correct simple data is prepared).
-   **Integration Tests**:
    -   Verify that `DiscoveryAgent` populates a local ChromaDB instance correctly when processing Last.fm data.
-   **Manual Verification**: Inspect the ChromaDB state after running the agent to ensure data is being added.

---

## 8. Success Criteria (MVP)

-   `DiscoveryAgent` successfully initializes a persistent ChromaDB client and collection.
-   Tracks processed by `DiscoveryAgent` from Last.fm are successfully added to ChromaDB using default embeddings on simple `"{artist} - {title}"` text.
-   The population process is robust and includes basic error handling.
-   The `DiscoveryAgent`'s primary recommendation capabilities (relying on Last.fm for MVP) remain functional.

---

## 9. Risks and Mitigation

-   **Risk**: Data Duplication if ID generation for `"{artist} - {title}"` is not robust.
    -   **Mitigation**: Careful normalization.
-   **Risk**: Growing ChromaDB Size.
    -   **Mitigation**: Accept for MVP. Future phases to consider management if it becomes an issue.
-   **Risk (Future)**: Transitioning from simple default embeddings to custom rich embeddings will require a strategy (e.g., re-embedding, new collection).
    -   **Mitigation**: Acknowledge as a future task. This MVP design does not lock us out of it.

---

## 10. Decisions & Future Evolution

Based on discussion, the following decisions guide this MVP integration:

-   **Persistence**: ChromaDB will use a **persistent client**. The path will be configurable via `AgentConfig` (or similar), defaulting to a local directory like `"./chroma_data"`.
-   **Granularity of "Document" for Embedding (MVP)**: For the initial MVP population, the `document` string to be embedded by ChromaDB's default model will be **`"{artist} - {title}"`**. This is for simplicity and to focus on the mechanics of population. This string will be enriched in a future phase when implementing the full custom embedding strategy.
-   **Configuration**: The ChromaDB path will be managed via `AgentConfig` (or a similar agent-specific configuration object).
-   **Primary Similarity (MVP)**: Last.fm's similarity functions will remain the primary mechanism for the `DiscoveryAgent`'s recommendations in the MVP.
-   **ChromaDB Querying (MVP)**: Querying ChromaDB for recommendations is optional and experimental for the MVP. The main goal is successful data population.

**Future Evolution (Post-MVP):**
-   Implement the custom embedding strategy (e.g., `genre + tags + mood + similar_artists` text for embeddings) as detailed in `beatdebate-design-doc.md`.
-   Transition `DiscoveryAgent` to use ChromaDB with these custom embeddings as the primary source for similarity search.
-   Refine querying strategies for ChromaDB.
-   Potentially re-embed tracks already in ChromaDB with the richer text representation or use a new collection.

---

This revised design clarifies the MVP scope for ChromaDB integration. Further discussion and refinement are welcome. 