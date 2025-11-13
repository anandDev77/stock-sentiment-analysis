"""
A/B testing framework for prompt optimization.

This module provides A/B testing capabilities to compare different
prompt variants and measure their effectiveness.
"""

import json
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

from ..config.settings import Settings, get_settings
from ..utils.logger import get_logger
from .cache import RedisCache

logger = get_logger(__name__)


class Variant(Enum):
    """A/B test variant types."""
    CONTROL = "control"
    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"


class ABTestingFramework:
    """
    A/B testing framework for prompt optimization.
    
    This framework allows testing different prompt variants and tracking
    their performance metrics to make data-driven decisions.
    
    Attributes:
        cache: Redis cache instance for storing test data
        active_tests: Dictionary of active A/B tests
        
    Example:
        >>> ab = ABTestingFramework(redis_cache=cache)
        >>> variant = ab.get_variant("sentiment_prompt")
        >>> result = analyze_with_variant(text, variant)
        >>> ab.track_result("sentiment_prompt", variant, result)
    """
    
    def __init__(self, redis_cache: Optional[RedisCache] = None):
        """
        Initialize A/B testing framework.
        
        Args:
            redis_cache: Redis cache instance for storing test data
        """
        self.cache = redis_cache
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        
        if not self.cache or not self.cache.client:
            logger.warning("Redis not available - A/B testing disabled")
            return
        
        logger.info("A/B testing framework initialized")
    
    def register_test(
        self,
        test_name: str,
        variants: List[str],
        traffic_split: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Register a new A/B test.
        
        Args:
            test_name: Name of the test
            variants: List of variant names (e.g., ["control", "variant_a"])
            traffic_split: Traffic split percentages (default: equal split)
            
        Returns:
            True if registered successfully
        """
        if not self.cache or not self.cache.client:
            return False
        
        if traffic_split is None:
            # Equal split
            split = 1.0 / len(variants)
            traffic_split = {v: split for v in variants}
        
        # Validate traffic split sums to 1.0
        total = sum(traffic_split.values())
        if abs(total - 1.0) > 0.01:
            logger.error(f"Traffic split must sum to 1.0, got {total}")
            return False
        
        test_config = {
            "variants": variants,
            "traffic_split": traffic_split,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        try:
            key = f"ab_test:config:{test_name}"
            self.cache.client.set(key, json.dumps(test_config))
            self.active_tests[test_name] = test_config
            logger.info(f"A/B test registered: {test_name} with variants {variants}")
            return True
        except Exception as e:
            logger.error(f"Error registering test: {e}")
            return False
    
    def get_variant(self, test_name: str, user_id: Optional[str] = None) -> str:
        """
        Get variant for a test based on traffic split.
        
        Args:
            test_name: Name of the test
            user_id: Optional user ID for consistent assignment
            
        Returns:
            Variant name
        """
        if not self.cache or not self.cache.client:
            return Variant.CONTROL.value
        
        # Load test config if not in memory
        if test_name not in self.active_tests:
            try:
                key = f"ab_test:config:{test_name}"
                config_data = self.cache.client.get(key)
                if config_data:
                    self.active_tests[test_name] = json.loads(config_data)
                else:
                    # Test not found, return control
                    return Variant.CONTROL.value
            except Exception as e:
                logger.error(f"Error loading test config: {e}")
                return Variant.CONTROL.value
        
        test_config = self.active_tests[test_name]
        variants = test_config.get("variants", [Variant.CONTROL.value])
        traffic_split = test_config.get("traffic_split", {})
        
        if not variants:
            return Variant.CONTROL.value
        
        # Use consistent assignment if user_id provided
        if user_id:
            # Hash user_id to get consistent variant
            import hashlib
            hash_val = int(hashlib.md5(f"{test_name}:{user_id}".encode()).hexdigest(), 16)
            rand = (hash_val % 10000) / 10000.0
        else:
            rand = random.random()
        
        # Assign variant based on traffic split
        cumulative = 0.0
        for variant, split in traffic_split.items():
            cumulative += split
            if rand <= cumulative:
                return variant
        
        # Fallback to first variant
        return variants[0]
    
    def track_result(
        self,
        test_name: str,
        variant: str,
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Track result metrics for a variant.
        
        Args:
            test_name: Name of the test
            variant: Variant name
            metrics: Dictionary of metrics to track
            
        Returns:
            True if tracked successfully
        """
        if not self.cache or not self.cache.client:
            return False
        
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            key = f"ab_test:results:{test_name}:{variant}:{today}"
            
            # Increment count
            self.cache.client.hincrby(key, "count", 1)
            
            # Store metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.cache.client.hincrbyfloat(key, f"metric:{metric_name}", metric_value)
                else:
                    # Store as JSON string
                    self.cache.client.hset(key, f"metric:{metric_name}", json.dumps(metric_value))
            
            # Set expiration (keep for 90 days)
            self.cache.client.expire(key, 86400 * 90)
            
            return True
        except Exception as e:
            logger.error(f"Error tracking result: {e}")
            return False
    
    def get_test_results(
        self,
        test_name: str,
        days: int = 7
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get aggregated test results.
        
        Args:
            test_name: Name of the test
            days: Number of days to aggregate
            
        Returns:
            Dictionary of variant -> metrics
        """
        if not self.cache or not self.cache.client:
            return {}
        
        results = {}
        
        try:
            # Get test config to know variants
            if test_name not in self.active_tests:
                key = f"ab_test:config:{test_name}"
                config_data = self.cache.client.get(key)
                if config_data:
                    self.active_tests[test_name] = json.loads(config_data)
            
            if test_name not in self.active_tests:
                return {}
            
            variants = self.active_tests[test_name].get("variants", [])
            
            # Aggregate results for each variant
            for variant in variants:
                variant_results = {
                    "count": 0,
                    "metrics": {}
                }
                
                # Aggregate across days
                for day_offset in range(days):
                    date = (datetime.now() - timedelta(days=day_offset)).strftime("%Y-%m-%d")
                    key = f"ab_test:results:{test_name}:{variant}:{date}"
                    
                    data = self.cache.client.hgetall(key)
                    if data:
                        variant_results["count"] += int(data.get("count", 0))
                        
                        # Aggregate metrics
                        for k, v in data.items():
                            if k.startswith("metric:"):
                                metric_name = k[7:]  # Remove "metric:" prefix
                                try:
                                    # Try to parse as number
                                    metric_value = float(v)
                                    if metric_name not in variant_results["metrics"]:
                                        variant_results["metrics"][metric_name] = 0.0
                                    variant_results["metrics"][metric_name] += metric_value
                                except ValueError:
                                    # Store as JSON
                                    variant_results["metrics"][metric_name] = json.loads(v)
                
                results[variant] = variant_results
            
            return results
        except Exception as e:
            logger.error(f"Error getting test results: {e}")
            return {}

