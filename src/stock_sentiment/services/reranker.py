"""
Re-ranking service using cross-encoder for improved precision.

Cross-encoders consider query-document interaction, providing more accurate
relevance scores than cosine similarity alone.
"""

from typing import List, Dict, Optional
from openai import AzureOpenAI

from ..config.settings import Settings, get_settings
from ..utils.logger import get_logger
from ..utils.retry import retry_with_exponential_backoff

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Re-rank search results using cross-encoder approach.
    
    Cross-encoders consider query-document interaction, providing more
    accurate relevance scores than bi-encoder (cosine similarity) approaches.
    
    This implementation uses Azure OpenAI for re-ranking, which provides
    better precision than simple cosine similarity.
    
    Attributes:
        client: Azure OpenAI client
        deployment_name: Model deployment name
        enabled: Whether re-ranking is enabled
        
    Example:
        >>> reranker = CrossEncoderReranker(settings=settings)
        >>> reranked = reranker.rerank(query, candidates, top_k=3)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        enabled: bool = True
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            settings: Application settings (optional)
            enabled: Whether re-ranking is enabled
        """
        self.settings = settings or get_settings()
        self.enabled = enabled
        
        if not enabled:
            logger.info("Cross-encoder reranking disabled")
            return
        
        if not self.settings.is_azure_openai_available():
            logger.warning("Azure OpenAI not available - reranking disabled")
            self.enabled = False
            return
        
        try:
            azure_config = self.settings.azure_openai
            self.client = AzureOpenAI(
                azure_endpoint=azure_config.endpoint,
                api_key=azure_config.api_key,
                api_version=azure_config.api_version
            )
            self.deployment_name = azure_config.deployment_name
            logger.info(f"Cross-encoder reranker initialized with {self.deployment_name}")
        except Exception as e:
            logger.error(f"Error initializing reranker: {e}")
            self.enabled = False
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 3
    ) -> List[Dict]:
        """
        Re-rank candidates using cross-encoder scoring.
        
        Args:
            query: Search query
            candidates: List of candidate documents with metadata
            top_k: Number of top results to return
            
        Returns:
            Re-ranked list of candidates
        """
        if not self.enabled or not candidates:
            return candidates[:top_k]
        
        if len(candidates) <= top_k:
            # No need to re-rank if we have fewer candidates than requested
            return candidates
        
        try:
            # Score each candidate using cross-encoder
            scored_candidates = []
            
            for candidate in candidates:
                # Extract text to score (title + summary)
                title = candidate.get('title', '')
                summary = candidate.get('summary', '')
                text = f"{title} {summary}".strip()
                
                if not text:
                    continue
                
                # Use Azure OpenAI to score relevance
                score = self._score_relevance(query, text)
                candidate['rerank_score'] = score
                scored_candidates.append(candidate)
            
            # Sort by rerank score (descending)
            scored_candidates.sort(
                reverse=True,
                key=lambda x: x.get('rerank_score', 0)
            )
            
            # Return top_k
            return scored_candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Fallback to original order
            return candidates[:top_k]
    
    @retry_with_exponential_backoff(max_attempts=2, initial_delay=1.0, max_delay=5.0)
    def _score_relevance(self, query: str, text: str) -> float:
        """
        Score relevance of text to query using cross-encoder.
        
        Args:
            query: Search query
            text: Document text to score
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        prompt = f"""Rate the relevance of the following text to the query on a scale of 0.0 to 1.0.

Query: {query}

Text: {text[:500]}

Respond with ONLY a number between 0.0 and 1.0 (e.g., 0.85). No explanation."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a relevance scorer. Respond with only a number between 0.0 and 1.0."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            # Extract number from response
            try:
                score = float(score_text)
                # Clamp to [0, 1]
                return max(0.0, min(1.0, score))
            except ValueError:
                logger.warning(f"Could not parse score: {score_text}")
                return 0.5  # Default neutral score
                
        except Exception as e:
            logger.error(f"Error scoring relevance: {e}")
            return 0.5  # Default neutral score

