# Design Doc: Underground Track Discovery Strategy

### 1. Problem Statement

Users want to discover "underground" or less popular tracks from a specific artist. For example, a user should be able to ask "Discover underground tracks by Kendrick Lamar" and get a list of his least popular, but still valid, songs. This requires a reliable way to fetch all tracks for an artist, filter out non-music items (like interludes or interviews), and then sort them by a popularity metric to surface the more obscure ones.

### 2. Current Implementation Analysis

The current implementation, as observed from `beatdebate.log`, has the following weaknesses:

- **A-1: Inaccurate Artist Matching:** The system incorrectly includes tracks where the artist is only featured (e.g., `Ab-Soul - Illuminate (feat. Kendrick Lamar)`) or where the artist name is part of a mashup (e.g., `KENDRICK LAMAR VS. LUPE FIASCO`).
- **A-2: Poor Track Filtering:** The results include many tracks titled `Unknown` or from non-canonical sources like mixtape websites. The current filtering that reduces 100 potential tracks to 33 "real" ones is not effective enough.
- **A-3: Noisy Data Sourcing:** The strategy of using multiple broad search queries (`rare`, `demo`, `underground`) introduces noisy and unstructured data, which complicates the filtering process.

### 3. Proposed Design

I propose a new, more streamlined two-stage process that prioritizes clean data sourcing and strict filtering.

#### 3.1. Stage 1: Comprehensive and Canonical Track Gathering

Instead of using heuristic-based text searches, we will use a canonical API endpoint to fetch an artist's tracks. The Last.fm `artist.getTopTracks` endpoint is a good candidate. We will need to handle pagination to retrieve all available tracks for a given artist. This will provide a cleaner and more structured dataset to work with from the start.

**Interface Idea:**

```python
from typing import List
from pydantic import BaseModel

class Track(BaseModel):
    name: str
    artist: str
    listeners: int
    # ... other relevant fields

def get_all_artist_tracks(artist_name: str) -> List[Track]:
    """Fetches all tracks for a given artist from a canonical source."""
    # ... implementation with pagination ...
```

#### 3.2. Stage 2: Strict Filtering and Ranking

Once we have the full list of tracks, we will apply a series of strict filters:

1.  **Artist Name Normalization and Matching:**
    -   Normalize both the queried artist name and the artist name from the track data (e.g., lowercase, remove whitespace).
    -   Perform an exact match between the normalized names.
    -   Explicitly filter out tracks where the artist name contains "feat.", "vs.", or is from a known mixtape source.

2.  **Track Validity Filter:**
    -   Filter out tracks with generic or invalid titles like "Unknown", "Untitled", or titles containing "skit", "interlude", "interview".
    -   **[Optional Enhancement]** For higher accuracy, we can verify tracks against another service like MusicBrainz or check for the existence of a Spotify ID to ensure they are legitimate songs.

3.  **Popularity Ranking:**
    -   Rank the remaining tracks by the `listeners` count in ascending order. Tracks with very few or zero listeners will appear first, fulfilling the "underground" requirement.

#### 3.3. Example Workflow (for "Kendrick Lamar")

1.  Call `get_all_artist_tracks("Kendrick Lamar")`.
2.  Receive a list of all his tracks from the canonical source.
3.  Filter this list:
    -   Keep only tracks where the primary artist is exactly "Kendrick Lamar".
    -   Discard `Ab-Soul - Illuminate (feat. Kendrick Lamar)`.
    -   Discard `KENDRICK LAMAR VS. LUPE FIASCO - Unknown`.
    -   Discard any track named `Unknown`.
4.  Sort the cleaned list by `listeners` (lowest to highest).
5.  Return the top N results.

### 4. Implementation and Testing Plan

We will implement this in two phases, staying on the current branch as per our established process.

-   **Phase 1: Core Logic Implementation**
    1.  Implement `get_all_artist_tracks` to fetch and paginate through all tracks for an artist from the chosen API.
    2.  Implement the strict filtering and ranking logic in a new module/service.
    3.  Integrate this new logic into the existing "Discover Artist" agentic workflow.

-   **Phase 2: Testing**
    1.  Write unit tests for the `get_all_artist_tracks` function and the filtering logic.
    2.  Write integration tests using artists like "Kendrick Lamar" and another artist with many collaborations (e.g., "Taylor Swift") to validate the end-to-end process.

This approach should significantly improve the quality and relevance of the "underground" tracks returned to the user. 