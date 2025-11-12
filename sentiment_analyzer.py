import os
from openai import AzureOpenAI
import json
from typing import Dict, List
import re
from textblob import TextBlob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with Azure OpenAI."""
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure OpenAI endpoint and API key must be set in .env file")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of given text using Azure OpenAI.
        Returns sentiment scores for positive, negative, and neutral.
        """
        prompt = f"""Analyze the sentiment of the following text about stocks/finance.
Return ONLY a valid JSON object with scores (0-1) for: positive, negative, neutral.
The scores should sum to approximately 1.0.

Text: "{text}"

Return only the JSON object, no other text:
"""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analyzer. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )

            # Extract JSON from response
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                positive = float(result.get('positive', 0))
                negative = float(result.get('negative', 0))
                neutral = float(result.get('neutral', 0))
                
                # Normalize to ensure they sum to 1
                total = positive + negative + neutral
                if total > 0:
                    positive = positive / total
                    negative = negative / total
                    neutral = neutral / total
                else:
                    # Fallback if all are 0
                    neutral = 1.0
                
                return {
                    'positive': positive,
                    'negative': negative,
                    'neutral': neutral
                }
            else:
                # Fallback to TextBlob for parsing errors
                return self._textblob_fallback(text)

        except Exception as e:
            print(f"Error analyzing sentiment with Azure OpenAI: {e}")
            # Fallback to TextBlob
            return self._textblob_fallback(text)
    
    def _textblob_fallback(self, text: str) -> Dict[str, float]:
        """Fallback sentiment analysis using TextBlob."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            return {'positive': min(polarity, 1.0), 'negative': 0, 'neutral': max(0, 1-polarity)}
        elif polarity < -0.1:
            return {'positive': 0, 'negative': min(abs(polarity), 1.0), 'neutral': max(0, 1-abs(polarity))}
        else:
            return {'positive': 0, 'negative': 0, 'neutral': 1}

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts."""
        return [self.analyze_sentiment(text) for text in texts]
