"""
Market News Service for PreciousAI
Provides real-time precious metals news with AI-powered analysis
"""

import os
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from openai import OpenAI
import json

logger = logging.getLogger(__name__)

class MarketNewsService:
    """Service for fetching and analyzing precious metals market news"""
    
    def __init__(self):
        # API Configuration
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not found. Using demo mode.")
            
        self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        
        # News categories and keywords
        self.metals_keywords = {
            'gold': ['gold', 'XAU', 'gold price', 'gold market', 'gold trading', 'gold investment'],
            'silver': ['silver', 'XAG', 'silver price', 'silver market', 'silver trading'],
            'platinum': ['platinum', 'XPT', 'platinum price', 'platinum market', 'platinum trading'],
            'palladium': ['palladium', 'XPD', 'palladium price', 'palladium market', 'palladium trading'],
            'general': ['precious metals', 'commodities', 'metals market', 'mining', 'bullion']
        }
        
        # News sources and topics
        self.news_topics = [
            'precious metals market analysis',
            'gold silver platinum palladium prices',
            'metals trading news today',
            'precious metals investment outlook',
            'central bank gold purchases',
            'mining industry news',
            'commodities market trends',
            'inflation precious metals hedge'
        ]
    
    def get_latest_news(self, metal: Optional[str] = None, limit: int = 10) -> Dict:
        """
        Get latest precious metals news with AI analysis
        
        Args:
            metal: Specific metal to focus on (gold, silver, platinum, palladium)
            limit: Number of news items to return
            
        Returns:
            Dict containing news articles with AI analysis
        """
        try:
            # Get news from Perplexity AI
            news_data = self._fetch_news_from_perplexity(metal, limit)
            
            # Analyze news with OpenAI
            analyzed_news = self._analyze_news_with_openai(news_data, metal)
            
            # Generate market summary
            market_summary = self._generate_market_summary(analyzed_news, metal)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'metal_focus': metal or 'all',
                'total_articles': len(analyzed_news),
                'market_summary': market_summary,
                'news_articles': analyzed_news,
                'last_updated': datetime.utcnow().isoformat(),
                'source': 'perplexity_openai_combined'
            }
            
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return self._get_demo_news(metal, limit)
    
    def _fetch_news_from_perplexity(self, metal: Optional[str], limit: int) -> List[Dict]:
        """Fetch news using Perplexity AI"""
        try:
            # Construct search query based on metal focus
            if metal and metal.lower() in self.metals_keywords:
                keywords = self.metals_keywords[metal.lower()]
                query = f"Latest news and market analysis for {metal} precious metal: {' OR '.join(keywords[:3])}"
            else:
                query = "Latest precious metals market news: gold silver platinum palladium prices analysis today"
            
            headers = {
                'Authorization': f'Bearer {self.perplexity_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'sonar-pro',
                'messages': [
                    {
                        'role': 'user',
                        'content': f"""Please provide the latest {limit} news items about {query}. 
                        
                        For each news item, provide:
                        1. Headline/Title
                        2. Brief summary (2-3 sentences)
                        3. Key price movements or market impacts
                        4. Source and timestamp if available
                        5. Relevance to precious metals investors
                        
                        Format as JSON array with fields: title, summary, impact, source, timestamp, relevance_score (1-10)"""
                    }
                ],
                'max_tokens': 2000,
                'temperature': 0.3
            }
            
            response = requests.post(
                'https://api.perplexity.ai/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Try to extract JSON from the response
                try:
                    # Look for JSON array in the response
                    import re
                    json_match = re.search(r'\[.*\]', content, re.DOTALL)
                    if json_match:
                        news_data = json.loads(json_match.group())
                        return news_data[:limit]
                except:
                    pass
                
                # If JSON parsing fails, parse the text response
                return self._parse_text_news_response(content, limit)
            else:
                logger.warning(f"Perplexity API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching news from Perplexity: {e}")
            return []
    
    def _parse_text_news_response(self, content: str, limit: int) -> List[Dict]:
        """Parse text response into structured news data"""
        news_items = []
        
        # Split content into sections and extract key information
        lines = content.split('\n')
        current_item = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for headlines (usually start with numbers or bullets)
            if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '•', '-']):
                if current_item:
                    news_items.append(current_item)
                    current_item = {}
                
                current_item = {
                    'title': line.split('.', 1)[-1].strip() if '.' in line else line,
                    'summary': '',
                    'impact': 'Market relevant',
                    'source': 'Market Analysis',
                    'timestamp': datetime.utcnow().isoformat(),
                    'relevance_score': 8
                }
            elif current_item and not current_item.get('summary'):
                current_item['summary'] = line
        
        if current_item:
            news_items.append(current_item)
        
        return news_items[:limit]
    
    def _analyze_news_with_openai(self, news_data: List[Dict], metal: Optional[str]) -> List[Dict]:
        """Analyze news articles with OpenAI for deeper insights"""
        if not news_data:
            return []
        
        if not self.openai_client:
            logger.warning("OpenAI client not available. Skipping analysis.")
            # Return articles with default analysis
            return [
                {
                    **article,
                    'ai_analysis': self._create_default_analysis(),
                    'analyzed_at': datetime.utcnow().isoformat()
                }
                for article in news_data
            ]
        
        try:
            analyzed_articles = []
            
            for article in news_data:
                # Analyze each article with OpenAI
                analysis_prompt = f"""
                Analyze this precious metals news article:
                
                Title: {article.get('title', 'N/A')}
                Summary: {article.get('summary', 'N/A')}
                
                Provide analysis in JSON format:
                {{
                    "sentiment": "bullish/bearish/neutral",
                    "price_impact": "positive/negative/neutral",
                    "key_factors": ["factor1", "factor2", "factor3"],
                    "investor_implications": "brief explanation",
                    "confidence_level": "high/medium/low",
                    "time_horizon": "short-term/medium-term/long-term"
                }}
                """
                
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "user", "content": analysis_prompt}
                        ],
                        max_tokens=300,
                        temperature=0.3
                    )
                    
                    analysis_content = response.choices[0].message.content
                    
                    # Try to parse JSON analysis
                    try:
                        import re
                        json_match = re.search(r'\{.*\}', analysis_content, re.DOTALL)
                        if json_match:
                            analysis = json.loads(json_match.group())
                        else:
                            analysis = self._create_default_analysis()
                    except:
                        analysis = self._create_default_analysis()
                    
                    # Combine original article with analysis
                    analyzed_article = {
                        **article,
                        'ai_analysis': analysis,
                        'analyzed_at': datetime.utcnow().isoformat()
                    }
                    
                    analyzed_articles.append(analyzed_article)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing article with OpenAI: {e}")
                    # Add article without analysis
                    analyzed_articles.append({
                        **article,
                        'ai_analysis': self._create_default_analysis(),
                        'analyzed_at': datetime.utcnow().isoformat()
                    })
            
            return analyzed_articles
            
        except Exception as e:
            logger.error(f"Error in OpenAI analysis: {e}")
            return news_data
    
    def _create_default_analysis(self) -> Dict:
        """Create default analysis when AI analysis fails"""
        return {
            'sentiment': 'neutral',
            'price_impact': 'neutral',
            'key_factors': ['Market dynamics', 'Economic conditions'],
            'investor_implications': 'Monitor market developments',
            'confidence_level': 'medium',
            'time_horizon': 'medium-term'
        }
    
    def _generate_market_summary(self, analyzed_news: List[Dict], metal: Optional[str]) -> Dict:
        """Generate overall market summary from analyzed news"""
        if not analyzed_news:
            return self._get_default_market_summary(metal)
        
        try:
            # Aggregate sentiment and impacts
            sentiments = [article.get('ai_analysis', {}).get('sentiment', 'neutral') for article in analyzed_news]
            impacts = [article.get('ai_analysis', {}).get('price_impact', 'neutral') for article in analyzed_news]
            
            # Count sentiments
            bullish_count = sentiments.count('bullish')
            bearish_count = sentiments.count('bearish')
            neutral_count = sentiments.count('neutral')
            
            # Determine overall sentiment
            if bullish_count > bearish_count:
                overall_sentiment = 'bullish'
            elif bearish_count > bullish_count:
                overall_sentiment = 'bearish'
            else:
                overall_sentiment = 'neutral'
            
            # Generate summary with OpenAI
            if not self.openai_client:
                summary_text = f"Market showing {overall_sentiment} sentiment based on {len(analyzed_news)} articles analyzed."
            else:
                summary_prompt = f"""
                Based on {len(analyzed_news)} recent precious metals news articles, create a market summary:
                
                Overall sentiment: {overall_sentiment}
                Bullish articles: {bullish_count}
                Bearish articles: {bearish_count}
                Neutral articles: {neutral_count}
                Focus: {metal or 'all precious metals'}
                
                Provide a concise 2-3 sentence market summary highlighting key trends and outlook.
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "user", "content": summary_prompt}
                    ],
                    max_tokens=150,
                    temperature=0.4
                )
                
                summary_text = response.choices[0].message.content
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_distribution': {
                    'bullish': bullish_count,
                    'bearish': bearish_count,
                    'neutral': neutral_count
                },
                'market_outlook': summary_text,
                'confidence_level': 'high' if max(bullish_count, bearish_count, neutral_count) > len(analyzed_news) * 0.6 else 'medium',
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating market summary: {e}")
            return self._get_default_market_summary(metal)
    
    def _get_default_market_summary(self, metal: Optional[str]) -> Dict:
        """Get default market summary when analysis fails"""
        return {
            'overall_sentiment': 'neutral',
            'sentiment_distribution': {
                'bullish': 0,
                'bearish': 0,
                'neutral': 1
            },
            'market_outlook': f'Monitoring {metal or "precious metals"} market developments. Stay informed with latest news and analysis.',
            'confidence_level': 'medium',
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def _get_demo_news(self, metal: Optional[str], limit: int) -> Dict:
        """Get demo news data when APIs are unavailable"""
        demo_articles = [
            {
                'title': f'{metal.title() if metal else "Gold"} Prices Show Resilience Amid Market Volatility',
                'summary': 'Precious metals continue to attract investor attention as a hedge against economic uncertainty and inflation concerns.',
                'impact': 'Positive for long-term investors',
                'source': 'Market Analysis',
                'timestamp': datetime.utcnow().isoformat(),
                'relevance_score': 9,
                'ai_analysis': {
                    'sentiment': 'bullish',
                    'price_impact': 'positive',
                    'key_factors': ['Economic uncertainty', 'Inflation hedge', 'Safe haven demand'],
                    'investor_implications': 'Consider precious metals allocation',
                    'confidence_level': 'high',
                    'time_horizon': 'long-term'
                }
            },
            {
                'title': 'Central Bank Purchases Support Precious Metals Demand',
                'summary': 'Global central banks continue to diversify reserves with precious metals purchases, providing fundamental support.',
                'impact': 'Structural demand support',
                'source': 'Financial News',
                'timestamp': (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                'relevance_score': 8,
                'ai_analysis': {
                    'sentiment': 'bullish',
                    'price_impact': 'positive',
                    'key_factors': ['Central bank demand', 'Reserve diversification', 'Institutional buying'],
                    'investor_implications': 'Strong fundamental support',
                    'confidence_level': 'high',
                    'time_horizon': 'medium-term'
                }
            }
        ]
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'metal_focus': metal or 'all',
            'total_articles': len(demo_articles),
            'market_summary': self._get_default_market_summary(metal),
            'news_articles': demo_articles[:limit],
            'last_updated': datetime.utcnow().isoformat(),
            'source': 'demo_data'
        }
    
    def get_news_by_category(self, category: str = 'general', limit: int = 5) -> Dict:
        """Get news by specific category"""
        return self.get_latest_news(metal=category if category != 'general' else None, limit=limit)
    
    def get_market_sentiment(self) -> Dict:
        """Get overall market sentiment analysis"""
        news_data = self.get_latest_news(limit=20)
        return news_data.get('market_summary', {})
    
    def search_news(self, query: str, limit: int = 10) -> Dict:
        """Search for specific news topics"""
        try:
            # Use Perplexity AI to search for specific topics
            headers = {
                'Authorization': f'Bearer {self.perplexity_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'sonar-pro',
                'messages': [
                    {
                        'role': 'user',
                        'content': f"Search for recent news about: {query} related to precious metals. Provide {limit} relevant articles with titles, summaries, and analysis."
                    }
                ],
                'max_tokens': 1500,
                'temperature': 0.3
            }
            
            response = requests.post(
                'https://api.perplexity.ai/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                return {
                    'query': query,
                    'results': self._parse_text_news_response(content, limit),
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                return {'query': query, 'results': [], 'error': 'Search failed'}
                
        except Exception as e:
            logger.error(f"Error searching news: {e}")
            return {'query': query, 'results': [], 'error': str(e)}

