"""
Cost tracking service for monitoring API usage and expenses.

This module tracks token usage and costs for Azure OpenAI API calls,
providing visibility into spending and enabling budget management.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from ..config.settings import Settings, get_settings
from ..utils.logger import get_logger
from .cache import RedisCache

logger = get_logger(__name__)


# Azure OpenAI pricing (as of 2024, update as needed)
# Prices per 1K tokens
PRICING = {
    "gpt-4": {
        "input": 0.03,   # $0.03 per 1K input tokens
        "output": 0.06,  # $0.06 per 1K output tokens
    },
    "gpt-4-turbo": {
        "input": 0.01,
        "output": 0.03,
    },
    "gpt-35-turbo": {
        "input": 0.0015,
        "output": 0.002,
    },
    "text-embedding-ada-002": {
        "input": 0.0001,  # $0.0001 per 1K tokens
        "output": 0.0,
    },
    "text-embedding-3-small": {
        "input": 0.00002,
        "output": 0.0,
    },
    "text-embedding-3-large": {
        "input": 0.00013,
        "output": 0.0,
    },
}


class CostTracker:
    """
    Track API costs for Azure OpenAI usage.
    
    This class provides cost tracking and reporting capabilities,
    storing daily costs in Redis for persistence and analysis.
    
    Attributes:
        cache: Redis cache instance for storing cost data
        settings: Application settings
        
    Example:
        >>> tracker = CostTracker(cache=redis_cache)
        >>> tracker.track_api_call("gpt-4", input_tokens=1000, output_tokens=500)
        >>> daily_costs = tracker.get_daily_costs()
    """
    
    def __init__(
        self,
        cache: Optional[RedisCache] = None,
        settings: Optional[Settings] = None
    ):
        """
        Initialize cost tracker.
        
        Args:
            cache: Redis cache instance (optional)
            settings: Application settings (optional)
        """
        self.cache = cache
        self.settings = settings or get_settings()
        self.cost_key_prefix = "costs"
    
    def _get_model_pricing(self, model: str) -> Dict[str, float]:
        """
        Get pricing for a model.
        
        Args:
            model: Model name (e.g., "gpt-4", "text-embedding-ada-002")
            
        Returns:
            Dictionary with input and output pricing per 1K tokens
        """
        # Normalize model name
        model_lower = model.lower()
        
        # Try exact match first
        if model_lower in PRICING:
            return PRICING[model_lower]
        
        # Try partial match (e.g., "gpt-4" matches "gpt-4-turbo")
        for key, pricing in PRICING.items():
            if key in model_lower or model_lower in key:
                return pricing
        
        # Default pricing (conservative estimate)
        logger.warning(f"Unknown model pricing for {model}, using default")
        return {"input": 0.01, "output": 0.02}
    
    def track_api_call(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        operation_type: str = "completion"
    ) -> float:
        """
        Track API call costs.
        
        Args:
            model: Model name used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation_type: Type of operation ("completion" or "embedding")
            
        Returns:
            Total cost in USD
        """
        if not self.cache or not self.cache.client:
            return 0.0
        
        pricing = self._get_model_pricing(model)
        
        # Calculate costs
        input_cost = (input_tokens / 1000.0) * pricing.get("input", 0.0)
        output_cost = (output_tokens / 1000.0) * pricing.get("output", 0.0)
        total_cost = input_cost + output_cost
        
        if total_cost == 0.0:
            return 0.0
        
        # Store in Redis with daily aggregation
        today = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # Track by model and operation type
            model_key = f"{self.cost_key_prefix}:daily:{today}:{model}:{operation_type}"
            
            # Increment cost
            self.cache.client.hincrbyfloat(model_key, "cost", total_cost)
            self.cache.client.hincrby(model_key, "input_tokens", input_tokens)
            self.cache.client.hincrby(model_key, "output_tokens", output_tokens)
            self.cache.client.hincrby(model_key, "calls", 1)
            
            # Set expiration (keep for 90 days)
            self.cache.client.expire(model_key, 86400 * 90)
            
            # Track total daily cost
            total_key = f"{self.cost_key_prefix}:daily:{today}:total"
            self.cache.client.hincrbyfloat(total_key, "cost", total_cost)
            self.cache.client.expire(total_key, 86400 * 90)
            
            
        except Exception as e:
            logger.error(f"Error tracking cost: {e}")
        
        return total_cost
    
    def get_daily_costs(self, date: Optional[str] = None) -> Dict[str, float]:
        """
        Get costs for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            Dictionary with cost breakdown by model and operation
        """
        if not self.cache or not self.cache.client:
            return {}
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # Get all keys for this date
            pattern = f"{self.cost_key_prefix}:daily:{date}:*"
            keys = []
            cursor = 0
            
            while True:
                cursor, partial_keys = self.cache.client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                keys.extend(partial_keys)
                if cursor == 0:
                    break
            
            costs = {}
            for key in keys:
                cost_data = self.cache.client.hgetall(key)
                if cost_data:
                    model_op = key.split(":")[-2:]  # Get model and operation
                    model_op_key = ":".join(model_op)
                    costs[model_op_key] = {
                        "cost": float(cost_data.get("cost", 0)),
                        "input_tokens": int(cost_data.get("input_tokens", 0)),
                        "output_tokens": int(cost_data.get("output_tokens", 0)),
                        "calls": int(cost_data.get("calls", 0))
                    }
            
            return costs
            
        except Exception as e:
            logger.error(f"Error getting daily costs: {e}")
            return {}
    
    def get_total_cost(self, days: int = 30) -> float:
        """
        Get total cost for the last N days.
        
        Args:
            days: Number of days to aggregate (default: 30)
            
        Returns:
            Total cost in USD
        """
        if not self.cache or not self.cache.client:
            return 0.0
        
        total = 0.0
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            total_key = f"{self.cost_key_prefix}:daily:{date}:total"
            
            try:
                cost_data = self.cache.client.hgetall(total_key)
                if cost_data:
                    total += float(cost_data.get("cost", 0))
            except Exception as e:
                logger.warning(f"Error getting cost for {date}: {e}")
        
        return total
    
    def get_cost_summary(self, days: int = 7) -> Dict:
        """
        Get cost summary for the last N days.
        
        Args:
            days: Number of days to summarize (default: 7)
            
        Returns:
            Dictionary with cost summary statistics
        """
        daily_costs = []
        total_cost = 0.0
        total_tokens = {"input": 0, "output": 0}
        total_calls = 0
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            costs = self.get_daily_costs(date)
            
            day_total = sum(item.get("cost", 0) for item in costs.values())
            daily_costs.append({"date": date, "cost": day_total})
            total_cost += day_total
            
            for item in costs.values():
                total_tokens["input"] += item.get("input_tokens", 0)
                total_tokens["output"] += item.get("output_tokens", 0)
                total_calls += item.get("calls", 0)
        
        return {
            "total_cost": total_cost,
            "average_daily_cost": total_cost / days if days > 0 else 0.0,
            "total_input_tokens": total_tokens["input"],
            "total_output_tokens": total_tokens["output"],
            "total_calls": total_calls,
            "daily_breakdown": daily_costs
        }

