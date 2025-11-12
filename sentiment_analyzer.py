import ollama
import json
from typing import Dict, List
import re
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self, model_name: str = "llama2:7b"):
        """Initialize the sentiment analyzer with Ollama model."""
        self.model_name = model_name
        self.client = ollama.Client()

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of given text using Ollama.
        Returns sentiment scores for positive, negative, and neutral.
        """
        prompt = f"""
        Analyze the sentiment of the following text about stocks/finance.
        Return ONLY a JSON object with scores (0-1) for: positive, negative, neutral.

        Text: "{text}"

        JSON:
        """

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )

            # Extract JSON from response
            content = response['response']
            json_match = re.search(r'\{.*\}', content, re.DOTALL)

            if json_match:
                result = json.loads(json_match.group())
                return {
                    'positive': float(result.get('positive', 0)),
                    'negative': float(result.get('negative', 0)),
                    'neutral': float(result.get('neutral', 0))
                }
            else:
                # Fallback to TextBlob for parsing errors
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity

                if polarity > 0.1:
                    return {'positive': polarity, 'negative': 0, 'neutral': 1-polarity}
                elif polarity < -0.1:
                    return {'positive': 0, 'negative': abs(polarity), 'neutral': 1-abs(polarity)}
                else:
                    return {'positive': 0, 'negative': 0, 'neutral': 1}

        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            # Fallback to TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                return {'positive': polarity, 'negative': 0, 'neutral': 1-polarity}
            elif polarity < -0.1:
                return {'positive': 0, 'negative': abs(polarity), 'neutral': 1-abs(polarity)}
            else:
                return {'positive': 0, 'negative': 0, 'neutral': 1}

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts."""
        return [self.analyze_sentiment(text) for text in texts]

