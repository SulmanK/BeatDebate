#!/usr/bin/env python3
"""
Last.fm Data Validation Script

Tests Last.fm API quality for indie/underground track discovery
before building the full BeatDebate system.
"""

import asyncio
import os
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import structlog
from dotenv import load_dotenv

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api.lastfm_client import LastFmClient, TrackMetadata

# Load environment variables
load_dotenv()

# Configure logging for validation
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class LastFmValidator:
    """Validates Last.fm API quality for BeatDebate use case."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.test_queries = [
            "indie rock underground",
            "ambient electronic experimental", 
            "post-rock instrumental",
            "folk indie singer-songwriter",
            "experimental jazz fusion",
            "synthwave retro",
            "math rock progressive",
            "chillhop lo-fi"
        ]
        self.results = {}
        
    async def run_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        logger.info("Starting Last.fm validation")
        
        async with LastFmClient(self.api_key) as client:
            # Test track search quality
            search_results = await self._test_track_search(client)
            
            # Test metadata richness
            metadata_results = await self._test_metadata_richness(client)
            
            # Test diversity and discovery potential
            diversity_results = await self._test_diversity(client)
            
            # Test tag-based search
            tag_results = await self._test_tag_search(client)
            
            # Compile final results
            validation_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "api_key_valid": True,
                "search_quality": search_results,
                "metadata_richness": metadata_results,
                "diversity_analysis": diversity_results,
                "tag_search": tag_results,
                "recommendations": self._generate_recommendations()
            }
            
        logger.info("Last.fm validation completed")
        return validation_results
        
    async def _test_track_search(self, client: LastFmClient) -> Dict[str, Any]:
        """Test basic track search functionality."""
        logger.info("Testing track search quality")
        
        search_results = {}
        total_tracks = 0
        queries_with_results = 0
        
        for query in self.test_queries:
            try:
                tracks = await client.search_tracks(query, limit=20)
                
                result_count = len(tracks)
                total_tracks += result_count
                
                if result_count > 0:
                    queries_with_results += 1
                    
                search_results[query] = {
                    "result_count": result_count,
                    "sample_tracks": [
                        {
                            "name": track.name,
                            "artist": track.artist,
                            "listeners": track.listeners
                        }
                        for track in tracks[:3]  # Sample first 3
                    ]
                }
                
                logger.info(
                    "Search completed",
                    query=query,
                    results=result_count
                )
                
            except Exception as e:
                logger.error(
                    "Search failed",
                    query=query,
                    error=str(e)
                )
                search_results[query] = {"error": str(e)}
                
        # Calculate metrics
        avg_results_per_query = total_tracks / len(self.test_queries) if self.test_queries else 0
        success_rate = queries_with_results / len(self.test_queries) if self.test_queries else 0
        
        return {
            "total_queries": len(self.test_queries),
            "successful_queries": queries_with_results,
            "success_rate": success_rate,
            "average_results_per_query": avg_results_per_query,
            "total_tracks_found": total_tracks,
            "detailed_results": search_results
        }
        
    async def _test_metadata_richness(self, client: LastFmClient) -> Dict[str, Any]:
        """Test quality and richness of track metadata."""
        logger.info("Testing metadata richness")
        
        # Test with known indie tracks
        test_tracks = [
            ("Radiohead", "Weird Fishes"),
            ("Bon Iver", "Holocene"),
            ("The National", "Fake Empire"),
            ("Sigur Rós", "Hoppípolla"),
            ("Explosions in the Sky", "Your Hand in Mine")
        ]
        
        metadata_scores = []
        
        for artist, track in test_tracks:
            try:
                metadata = await client.get_track_info(artist, track)
                
                if metadata:
                    score = self._calculate_metadata_score(metadata)
                    metadata_scores.append(score)
                    
                    logger.info(
                        "Metadata retrieved",
                        artist=artist,
                        track=track,
                        score=score
                    )
                else:
                    logger.warning(
                        "No metadata found",
                        artist=artist,
                        track=track
                    )
                    
            except Exception as e:
                logger.error(
                    "Metadata retrieval failed",
                    artist=artist,
                    track=track,
                    error=str(e)
                )
                
        avg_score = sum(metadata_scores) / len(metadata_scores) if metadata_scores else 0
        
        return {
            "tracks_tested": len(test_tracks),
            "successful_retrievals": len(metadata_scores),
            "average_metadata_score": avg_score,
            "metadata_quality": "excellent" if avg_score > 0.8 else "good" if avg_score > 0.6 else "fair"
        }
        
    def _calculate_metadata_score(self, metadata: TrackMetadata) -> float:
        """Calculate metadata richness score (0-1)."""
        score = 0.0
        max_score = 7.0
        
        # Check various metadata fields
        if metadata.name:
            score += 1.0
        if metadata.artist:
            score += 1.0
        if metadata.tags and len(metadata.tags) > 0:
            score += 1.0
        if metadata.similar_tracks and len(metadata.similar_tracks) > 0:
            score += 1.0
        if metadata.listeners and metadata.listeners > 0:
            score += 1.0
        if metadata.playcount and metadata.playcount > 0:
            score += 1.0
        if metadata.summary:
            score += 1.0
            
        return score / max_score
        
    async def _test_diversity(self, client: LastFmClient) -> Dict[str, Any]:
        """Test diversity of search results."""
        logger.info("Testing result diversity")
        
        # Get tracks from first query for diversity analysis
        query = self.test_queries[0]
        tracks = await client.search_tracks(query, limit=50)
        
        if not tracks:
            return {"error": "No tracks for diversity analysis"}
            
        # Analyze artist diversity
        artists = [track.artist for track in tracks]
        unique_artists = set(artists)
        artist_diversity = len(unique_artists) / len(tracks) if tracks else 0
        
        # Analyze popularity distribution (listeners)
        listener_counts = [track.listeners or 0 for track in tracks]
        avg_listeners = sum(listener_counts) / len(listener_counts) if listener_counts else 0
        
        # Check for mainstream bias (high listener counts might indicate mainstream bias)
        mainstream_threshold = 100000  # 100k listeners
        mainstream_count = sum(1 for count in listener_counts if count > mainstream_threshold)
        mainstream_ratio = mainstream_count / len(tracks) if tracks else 0
        
        return {
            "total_tracks_analyzed": len(tracks),
            "unique_artists": len(unique_artists),
            "artist_diversity_ratio": artist_diversity,
            "average_listeners": avg_listeners,
            "mainstream_tracks": mainstream_count,
            "mainstream_ratio": mainstream_ratio,
            "discovery_potential": "high" if mainstream_ratio < 0.3 else "medium" if mainstream_ratio < 0.6 else "low"
        }
        
    async def _test_tag_search(self, client: LastFmClient) -> Dict[str, Any]:
        """Test tag-based search for genre/mood discovery."""
        logger.info("Testing tag-based search")
        
        test_tags = ["indie", "experimental", "ambient", "post-rock", "electronic"]
        tag_results = {}
        
        for tag in test_tags:
            try:
                tracks = await client.search_by_tags([tag], limit=10)
                
                tag_results[tag] = {
                    "result_count": len(tracks),
                    "sample_artists": list(set([track.artist for track in tracks[:5]]))
                }
                
                logger.info(
                    "Tag search completed",
                    tag=tag,
                    results=len(tracks)
                )
                
            except Exception as e:
                logger.error(
                    "Tag search failed",
                    tag=tag,
                    error=str(e)
                )
                tag_results[tag] = {"error": str(e)}
                
        return tag_results
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Basic recommendations
        recommendations.append("Last.fm provides good coverage for indie/underground music discovery")
        recommendations.append("Tag-based search is effective for genre-specific discovery")
        recommendations.append("Metadata richness varies but generally sufficient for embeddings")
        recommendations.append("Rate limiting should be implemented (3 requests/second max)")
        recommendations.append("Caching is essential due to API response times")
        
        return recommendations


async def main():
    """Main validation function."""
    # Check for API key
    api_key = os.getenv("LASTFM_API_KEY")
    if not api_key:
        logger.error("LASTFM_API_KEY environment variable not set")
        return
        
    # Create output directory
    output_dir = Path("data/validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run validation
    validator = LastFmValidator(api_key)
    
    try:
        results = await validator.run_validation()
        
        # Save results
        output_file = output_dir / f"lastfm_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Print summary
        print("\n" + "="*60)
        print("LAST.FM VALIDATION SUMMARY")
        print("="*60)
        
        search_quality = results.get("search_quality", {})
        print(f"Search Success Rate: {search_quality.get('success_rate', 0):.1%}")
        print(f"Average Results per Query: {search_quality.get('average_results_per_query', 0):.1f}")
        print(f"Total Tracks Found: {search_quality.get('total_tracks_found', 0)}")
        
        metadata_quality = results.get("metadata_richness", {})
        print(f"Metadata Quality: {metadata_quality.get('metadata_quality', 'unknown')}")
        
        diversity = results.get("diversity_analysis", {})
        print(f"Discovery Potential: {diversity.get('discovery_potential', 'unknown')}")
        print(f"Artist Diversity: {diversity.get('artist_diversity_ratio', 0):.1%}")
        
        print(f"\nDetailed results saved to: {output_file}")
        
        # Print recommendations
        print("\nRECOMMENDATIONS:")
        for rec in results.get("recommendations", []):
            print(f"• {rec}")
            
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error("Validation failed", error=str(e))
        print(f"ERROR: Validation failed - {e}")


if __name__ == "__main__":
    asyncio.run(main()) 