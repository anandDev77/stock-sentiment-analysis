"""
Multi-tier caching implementation for optimal performance.

This module provides a three-tier caching strategy:
- L1: In-memory cache (fastest, smallest capacity)
- L2: Redis cache (fast, medium capacity)
- L3: Disk cache (slowest, largest capacity)

Industry best practice: Reduces latency by serving from fastest tier available.
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime, timedelta

from ..config.settings import Settings, get_settings
from ..utils.logger import get_logger
from .cache import RedisCache

logger = get_logger(__name__)


class MultiTierCache:
    """
    Multi-tier cache implementation for optimal performance.
    
    Cache hierarchy:
    1. L1 (Memory): Fastest access, limited capacity, process-local
    2. L2 (Redis): Fast access, medium capacity, shared across processes
    3. L3 (Disk): Slower access, largest capacity, persistent
    
    Attributes:
        l1_cache: In-memory dictionary cache
        l2_cache: Redis cache instance
        l3_cache_dir: Directory for disk cache
        l1_max_size: Maximum items in L1 cache
        l1_ttl: TTL for L1 cache items (seconds)
        
    Example:
        >>> cache = MultiTierCache(redis_cache=redis_cache)
        >>> cache.set("key", {"data": "value"}, ttl=3600)
        >>> value = cache.get("key")  # Checks L1 -> L2 -> L3
    """
    
    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
        l1_max_size: int = 1000,
        l1_ttl: int = 300,  # 5 minutes
        l3_cache_dir: Optional[Path] = None
    ):
        """
        Initialize multi-tier cache.
        
        Args:
            redis_cache: Redis cache instance for L2 (optional)
            l1_max_size: Maximum number of items in L1 cache
            l1_ttl: TTL for L1 cache items in seconds
            l3_cache_dir: Directory for L3 disk cache (optional)
        """
        self.l1_cache: Dict[str, Dict[str, Any]] = {}
        self.l2_cache = redis_cache
        self.l1_max_size = l1_max_size
        self.l1_ttl = l1_ttl
        
        # Track last cache tier used (for UI display)
        self.last_tier_used: Optional[str] = None
        self.tier_stats = {'l1_hits': 0, 'l2_hits': 0, 'l3_hits': 0, 'misses': 0}
        
        # L3 disk cache directory
        if l3_cache_dir:
            self.l3_cache_dir = Path(l3_cache_dir)
            self.l3_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Default to temp directory
            import tempfile
            self.l3_cache_dir = Path(tempfile.gettempdir()) / "stock_sentiment_cache"
            self.l3_cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Multi-tier cache initialized: "
            f"L1 (memory, max={l1_max_size}), "
            f"L2 (Redis: {'enabled' if redis_cache and redis_cache.client else 'disabled'}), "
            f"L3 (disk: {self.l3_cache_dir})"
        )
    
    def _get_l3_path(self, key: str) -> Path:
        """Get file path for L3 cache key."""
        # Create hash-based subdirectory to avoid too many files in one directory
        key_hash = hashlib.md5(key.encode()).hexdigest()
        subdir = self.l3_cache_dir / key_hash[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{key_hash}.cache"
    
    def _clean_l1_cache(self):
        """Remove expired items and enforce size limit."""
        now = datetime.now()
        
        # Remove expired items
        expired_keys = [
            k for k, v in self.l1_cache.items()
            if now > v.get('expires_at', datetime.min)
        ]
        for key in expired_keys:
            del self.l1_cache[key]
        
        # Enforce size limit (LRU: remove oldest)
        if len(self.l1_cache) > self.l1_max_size:
            # Sort by access time and remove oldest
            sorted_items = sorted(
                self.l1_cache.items(),
                key=lambda x: x[1].get('accessed_at', datetime.min)
            )
            items_to_remove = len(self.l1_cache) - self.l1_max_size
            for key, _ in sorted_items[:items_to_remove]:
                del self.l1_cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache, checking L1 -> L2 -> L3.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # Check L1 (memory)
        if key in self.l1_cache:
            item = self.l1_cache[key]
            if datetime.now() <= item.get('expires_at', datetime.max):
                # Update access time
                item['accessed_at'] = datetime.now()
                self.last_tier_used = "L1 (Memory)"
                self.tier_stats['l1_hits'] += 1
                return item['value']
            else:
                # Expired, remove from L1
                del self.l1_cache[key]
        
        # Check L2 (Redis)
        if self.l2_cache and self.l2_cache.client:
            value = self.l2_cache.get(key)
            if value is not None:
                # Populate L1 with value from L2
                self._set_l1(key, value)
                self.last_tier_used = "L2 (Redis)"
                self.tier_stats['l2_hits'] += 1
                return value
        
        # Check L3 (disk)
        l3_path = self._get_l3_path(key)
        if l3_path.exists():
            try:
                with open(l3_path, 'rb') as f:
                    item = pickle.load(f)
                    expires_at = item.get('expires_at')
                    if expires_at and datetime.now() <= expires_at:
                        value = item['value']
                        # Populate L2 and L1
                        if self.l2_cache:
                            self.l2_cache.set(key, value, ttl=3600)
                        self._set_l1(key, value)
                        self.last_tier_used = "L3 (Disk)"
                        self.tier_stats['l3_hits'] += 1
                        return value
                    else:
                        # Expired, remove file
                        l3_path.unlink()
            except Exception as e:
                logger.warning(f"Error reading L3 cache: {e}")
        
        # Cache miss
        self.last_tier_used = "MISS"
        self.tier_stats['misses'] += 1
        return None
    
    def _set_l1(self, key: str, value: Any):
        """Set value in L1 cache."""
        self._clean_l1_cache()
        self.l1_cache[key] = {
            'value': value,
            'expires_at': datetime.now() + timedelta(seconds=self.l1_ttl),
            'accessed_at': datetime.now()
        }
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set value in all cache tiers.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if set successfully
        """
        try:
            # Set in L1
            self._set_l1(key, value)
            
            # Set in L2 (Redis)
            if self.l2_cache and self.l2_cache.client:
                self.l2_cache.set(key, value, ttl=ttl)
            
            # Set in L3 (disk)
            l3_path = self._get_l3_path(key)
            try:
                item = {
                    'value': value,
                    'expires_at': datetime.now() + timedelta(seconds=ttl),
                    'created_at': datetime.now()
                }
                with open(l3_path, 'wb') as f:
                    pickle.dump(item, f)
            except Exception as e:
                logger.warning(f"Error writing L3 cache: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    def clear_l1(self):
        """Clear L1 (memory) cache."""
        self.l1_cache.clear()
        logger.info("L1 cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        l3_count = sum(1 for _ in self.l3_cache_dir.rglob("*.cache"))
        
        total_hits = sum([
            self.tier_stats['l1_hits'],
            self.tier_stats['l2_hits'],
            self.tier_stats['l3_hits']
        ])
        total_requests = total_hits + self.tier_stats['misses']
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'l1_size': len(self.l1_cache),
            'l1_max_size': self.l1_max_size,
            'l2_enabled': self.l2_cache is not None and self.l2_cache.client is not None,
            'l3_count': l3_count,
            'l3_directory': str(self.l3_cache_dir),
            'last_tier_used': self.last_tier_used or "None",
            'tier_stats': self.tier_stats.copy(),
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

