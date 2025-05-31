"""
Cache Management System

Provides file-based caching with TTL for API responses and track metadata.
Optimized for BeatDebate's specific caching needs.
"""

import json
import hashlib
import time
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import asdict, is_dataclass

import structlog
from diskcache import Cache

logger = structlog.get_logger(__name__)


class CacheManager:
    """
    File-based cache manager with TTL support.
    
    Handles caching for:
    - Last.fm API responses
    - Spotify API responses  
    - Track metadata
    - Agent strategies
    - User preferences
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize diskcache for different data types
        self.caches = {
            "lastfm": Cache(str(self.cache_dir / "lastfm")),
            "spotify": Cache(str(self.cache_dir / "spotify")),
            "tracks": Cache(str(self.cache_dir / "tracks")),
            "strategies": Cache(str(self.cache_dir / "strategies")),
            "preferences": Cache(str(self.cache_dir / "preferences")),
            "embeddings": Cache(str(self.cache_dir / "embeddings"))
        }
        
        # Default TTL values (in seconds)
        self.default_ttl = {
            "lastfm": 7 * 24 * 3600,      # 1 week
            "spotify": 7 * 24 * 3600,     # 1 week
            "tracks": 3 * 24 * 3600,      # 3 days
            "strategies": 12 * 3600,      # 12 hours
            "preferences": 24 * 3600,     # 1 day
            "embeddings": 30 * 24 * 3600, # 30 days
        }
        
        logger.info(
            "Cache manager initialized",
            cache_dir=str(self.cache_dir),
            cache_types=list(self.caches.keys())
        )
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for caching."""
        if is_dataclass(value):
            return asdict(value)
        elif isinstance(value, list) and value and is_dataclass(value[0]):
            return [asdict(item) for item in value]
        return value
    
    def _deserialize_track_metadata(self, data: Dict[str, Any]) -> Any:
        """Deserialize track metadata from cached dictionary."""
        try:
            # Import here to avoid circular imports
            from ..models.metadata_models import UnifiedTrackMetadata, MetadataSource
            from datetime import datetime
            
            # Handle datetime field
            if 'last_updated' in data and isinstance(data['last_updated'], str):
                data['last_updated'] = datetime.fromisoformat(data['last_updated'])
            
            # Handle MetadataSource enum
            if 'source' in data and isinstance(data['source'], str):
                data['source'] = MetadataSource(data['source'])
            
            # Create UnifiedTrackMetadata object
            return UnifiedTrackMetadata(**data)
            
        except Exception as e:
            logger.warning(f"Failed to deserialize track metadata: {e}")
            return data  # Return as dict if deserialization fails
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(
        self, 
        cache_type: str, 
        key: str, 
        default: Any = None
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            cache_type: Type of cache (lastfm, spotify, etc.)
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        if cache_type not in self.caches:
            logger.warning("Invalid cache type", cache_type=cache_type)
            return default
            
        try:
            cache = self.caches[cache_type]
            value = cache.get(key, default)
            
            if value is not default:
                logger.debug(
                    "Cache hit",
                    cache_type=cache_type,
                    key=key[:16] + "..."
                )
            else:
                logger.debug(
                    "Cache miss",
                    cache_type=cache_type,
                    key=key[:16] + "..."
                )
                
            return value
            
        except Exception as e:
            logger.error(
                "Cache get failed",
                cache_type=cache_type,
                key=key[:16] + "...",
                error=str(e)
            )
            return default
    
    def set(
        self, 
        cache_type: str, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            cache_type: Type of cache
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful
        """
        if cache_type not in self.caches:
            logger.warning("Invalid cache type", cache_type=cache_type)
            return False
            
        try:
            cache = self.caches[cache_type]
            if ttl is None:
                ttl = self.default_ttl.get(cache_type, 3600)
                
            # Serialize dataclasses
            serialized_value = self._serialize_value(value)
            
            cache.set(key, serialized_value, expire=ttl)
            
            logger.debug(
                "Cache set",
                cache_type=cache_type,
                key=key[:16] + "...",
                ttl=ttl
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Cache set failed",
                cache_type=cache_type,
                key=key[:16] + "...",
                error=str(e)
            )
            return False
    
    def delete(self, cache_type: str, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            cache_type: Type of cache
            key: Cache key
            
        Returns:
            True if successful
        """
        if cache_type not in self.caches:
            return False
            
        try:
            cache = self.caches[cache_type]
            return cache.delete(key)
            
        except Exception as e:
            logger.error(
                "Cache delete failed",
                cache_type=cache_type,
                key=key[:16] + "...",
                error=str(e)
            )
            return False
    
    def clear(self, cache_type: Optional[str] = None) -> bool:
        """
        Clear cache(s).
        
        Args:
            cache_type: Specific cache to clear (all if None)
            
        Returns:
            True if successful
        """
        try:
            if cache_type:
                if cache_type in self.caches:
                    self.caches[cache_type].clear()
                    logger.info("Cache cleared", cache_type=cache_type)
                    return True
                else:
                    logger.warning("Invalid cache type", cache_type=cache_type)
                    return False
            else:
                for cache_name, cache in self.caches.items():
                    cache.clear()
                logger.info("All caches cleared")
                return True
                
        except Exception as e:
            logger.error("Cache clear failed", error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        stats = {}
        
        for cache_name, cache in self.caches.items():
            try:
                # Get basic stats
                cache_stats = {
                    "size": len(cache),
                    "volume": cache.volume(),
                    "directory": str(cache.directory)
                }
                
                stats[cache_name] = cache_stats
                
            except Exception as e:
                logger.error(
                    "Failed to get cache stats",
                    cache_name=cache_name,
                    error=str(e)
                )
                stats[cache_name] = {"error": str(e)}
                
        return stats
    
    def warm_cache(self, cache_type: str, warm_data: Dict[str, Any]) -> int:
        """
        Warm cache with pre-computed data.
        
        Args:
            cache_type: Type of cache to warm
            warm_data: Dictionary of key-value pairs
            
        Returns:
            Number of items cached
        """
        if cache_type not in self.caches:
            logger.warning("Invalid cache type", cache_type=cache_type)
            return 0
            
        cached_count = 0
        
        for key, value in warm_data.items():
            if self.set(cache_type, key, value):
                cached_count += 1
                
        logger.info(
            "Cache warmed",
            cache_type=cache_type,
            items_cached=cached_count,
            total_items=len(warm_data)
        )
        
        return cached_count
    
    # Convenience methods for specific cache types
    
    def cache_lastfm_response(
        self, 
        method: str, 
        params: Dict[str, Any], 
        response: Any
    ) -> str:
        """Cache Last.fm API response."""
        key = self._generate_key("lastfm", method, **params)
        self.set("lastfm", key, response)
        return key
    
    def get_lastfm_response(
        self, 
        method: str, 
        params: Dict[str, Any]
    ) -> Any:
        """Get cached Last.fm API response."""
        key = self._generate_key("lastfm", method, **params)
        return self.get("lastfm", key)
    
    def cache_spotify_response(
        self, 
        endpoint: str, 
        params: Dict[str, Any], 
        response: Any
    ) -> str:
        """Cache Spotify API response."""
        key = self._generate_key("spotify", endpoint, **params)
        self.set("spotify", key, response)
        return key
    
    def get_spotify_response(
        self, 
        endpoint: str, 
        params: Dict[str, Any]
    ) -> Any:
        """Get cached Spotify API response."""
        key = self._generate_key("spotify", endpoint, **params)
        return self.get("spotify", key)
    
    def cache_track_metadata(
        self, 
        artist: str, 
        track: str, 
        metadata: Any
    ) -> str:
        """Cache track metadata."""
        key = self._generate_key("track", artist.lower(), track.lower())
        self.set("tracks", key, metadata)
        return key
    
    def get_track_metadata(self, artist: str, track: str) -> Any:
        """Get cached track metadata."""
        key = self._generate_key("track", artist.lower(), track.lower())
        cached_data = self.get("tracks", key)
        
        # If we got cached data and it's a dictionary, try to deserialize it
        if cached_data is not None and isinstance(cached_data, dict):
            return self._deserialize_track_metadata(cached_data)
        
        return cached_data
    
    def cache_strategy(
        self, 
        user_query: str, 
        strategy: Dict[str, Any]
    ) -> str:
        """Cache planning strategy."""
        key = self._generate_key("strategy", user_query)
        self.set("strategies", key, strategy, ttl=12 * 3600)  # 12 hours
        return key
    
    def get_strategy(self, user_query: str) -> Optional[Dict[str, Any]]:
        """Get cached planning strategy."""
        key = self._generate_key("strategy", user_query)
        return self.get("strategies", key)
    
    def cache_user_preferences(
        self, 
        user_id: str, 
        preferences: Dict[str, Any]
    ) -> str:
        """Cache user preferences."""
        key = self._generate_key("user_prefs", user_id)
        self.set("preferences", key, preferences)
        return key
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user preferences."""
        key = self._generate_key("user_prefs", user_id)
        return self.get("preferences", key)
    
    def cache_embeddings(
        self, 
        text_hash: str, 
        embeddings: List[float]
    ) -> str:
        """Cache text embeddings."""
        self.set("embeddings", text_hash, embeddings, ttl=30 * 24 * 3600)
        return text_hash
    
    def get_embeddings(self, text_hash: str) -> Optional[List[float]]:
        """Get cached text embeddings."""
        return self.get("embeddings", text_hash)
    
    def close(self) -> None:
        """Close all cache connections."""
        for cache_name, cache in self.caches.items():
            try:
                cache.close()
                logger.debug("Cache closed", cache_name=cache_name)
            except Exception as e:
                logger.error(
                    "Failed to close cache",
                    cache_name=cache_name,
                    error=str(e)
                )


# Global cache instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        cache_dir = os.getenv("CACHE_DIR", "data/cache")
        _cache_manager = CacheManager(cache_dir)
    return _cache_manager


def close_cache_manager() -> None:
    """Close global cache manager."""
    global _cache_manager
    if _cache_manager:
        _cache_manager.close()
        _cache_manager = None 