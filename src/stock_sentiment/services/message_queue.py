"""
Message queue implementation using Redis Streams for async processing.

This module provides async job processing capabilities using Redis Streams,
allowing non-blocking sentiment analysis and data collection.
"""

import json
import time
from typing import Dict, Optional, Callable, Any, List
from datetime import datetime

from ..config.settings import Settings, get_settings
from ..utils.logger import get_logger
from .cache import RedisCache

logger = get_logger(__name__)


class MessageQueue:
    """
    Message queue using Redis Streams for async job processing.
    
    This enables non-blocking processing of sentiment analysis jobs,
    improving UI responsiveness and scalability.
    
    Attributes:
        cache: Redis cache instance
        stream_name: Name of the Redis stream
        consumer_group: Consumer group name
        consumer_name: Consumer name
        
    Example:
        >>> queue = MessageQueue(redis_cache=cache)
        >>> queue.enqueue("sentiment", {"text": "...", "symbol": "AAPL"})
        >>> result = queue.process_job(process_sentiment)
    """
    
    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
        stream_name: str = "sentiment:jobs",
        consumer_group: str = "sentiment_workers",
        consumer_name: Optional[str] = None
    ):
        """
        Initialize message queue.
        
        Args:
            redis_cache: Redis cache instance
            stream_name: Name of the Redis stream
            consumer_group: Consumer group name
            consumer_name: Consumer name (auto-generated if None)
        """
        self.cache = redis_cache
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name or f"worker_{int(time.time())}"
        
        if not self.cache or not self.cache.client:
            logger.warning("Redis not available - message queue disabled")
            return
        
        # Create consumer group if it doesn't exist
        try:
            self.cache.client.xgroup_create(
                name=stream_name,
                groupname=consumer_group,
                id="0",
                mkstream=True
            )
            logger.info(f"Created consumer group: {consumer_group}")
        except Exception as e:
            # Group might already exist
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Error creating consumer group: {e}")
    
    def enqueue(
        self,
        job_type: str,
        job_data: Dict[str, Any],
        priority: int = 0
    ) -> Optional[str]:
        """
        Enqueue a job for async processing.
        
        Args:
            job_type: Type of job (e.g., "sentiment", "collect")
            job_data: Job data dictionary
            priority: Job priority (higher = more important)
            
        Returns:
            Job ID or None if failed
        """
        if not self.cache or not self.cache.client:
            logger.warning("Cannot enqueue job - Redis not available")
            return None
        
        try:
            job = {
                "type": job_type,
                "data": json.dumps(job_data),
                "priority": str(priority),
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            }
            
            job_id = self.cache.client.xadd(
                self.stream_name,
                job,
                maxlen=10000  # Keep last 10k jobs
            )
            
            logger.info(f"Job enqueued: {job_type} (ID: {job_id})")
            return job_id
        except Exception as e:
            logger.error(f"Error enqueueing job: {e}")
            return None
    
    def dequeue(
        self,
        count: int = 1,
        block: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Dequeue jobs from the stream.
        
        Args:
            count: Number of jobs to dequeue
            block: Block time in milliseconds (0 = non-blocking)
            
        Returns:
            List of job dictionaries
        """
        if not self.cache or not self.cache.client:
            return []
        
        try:
            # Read from stream using consumer group
            messages = self.cache.client.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams={self.stream_name: ">"},
                count=count,
                block=block
            )
            
            jobs = []
            for stream, msgs in messages:
                for msg_id, msg_data in msgs:
                    try:
                        job_data = json.loads(msg_data.get("data", "{}"))
                        jobs.append({
                            "id": msg_id,
                            "type": msg_data.get("type"),
                            "data": job_data,
                            "priority": int(msg_data.get("priority", 0))
                        })
                    except Exception as e:
                        logger.error(f"Error parsing job: {e}")
                        # Acknowledge bad job to remove from stream
                        self.acknowledge(msg_id)
            
            return jobs
        except Exception as e:
            logger.error(f"Error dequeueing jobs: {e}")
            return []
    
    def acknowledge(self, job_id: str) -> bool:
        """
        Acknowledge job completion.
        
        Args:
            job_id: Job ID to acknowledge
            
        Returns:
            True if acknowledged successfully
        """
        if not self.cache or not self.cache.client:
            return False
        
        try:
            self.cache.client.xack(
                self.stream_name,
                self.consumer_group,
                job_id
            )
            return True
        except Exception as e:
            logger.error(f"Error acknowledging job: {e}")
            return False
    
    def get_queue_length(self) -> int:
        """Get number of pending jobs in queue."""
        if not self.cache or not self.cache.client:
            return 0
        
        try:
            return self.cache.client.xlen(self.stream_name)
        except Exception:
            return 0

