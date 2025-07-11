"""
Met-Ask Service for PreciousAI
Intelligent Q&A system for precious metals questions
"""

import os
import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional
from openai import OpenAI
import json
import re

logger = logging.getLogger(__name__)

class MetAskService:
    """Intelligent Q&A service for precious metals questions"""
    
    def __init__(self):
        # API Configuration
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not found. Using demo mode.")
            
        self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        
        # Knowledge base categories
        self.knowledge_categories = {
            'pricing': ['price', 'cost', 'value', 'worth', 'expensive', 'cheap', 'market'],
            'investment': ['invest', 'portfolio', 'buy', 'sell', 'return', 'profit', 'loss'],
            'technical': ['purity', 'karat', 'fineness', 'alloy', 'composition', 'properties'],
            'market': ['trend', 'forecast', 'outlook', 'analysis', 'prediction', 'future'],
            'storage': ['store', 'vault', 'security', 'insurance', 'custody', 'safe'],
            'trading': ['trade', 'exchange', 'broker', 'dealer', 'spread', 'commission'],
            'history': ['historical', 'past', 'ancient', 'origin', 'discovery', 'timeline'],
            'industrial': ['industrial', 'manufacturing', 'electronics', 'automotive', 'medical'],
            'comparison': ['compare', 'difference', 'better', 'versus', 'vs', 'which'],
            'beginner': ['beginner', 'start', 'basics', 'introduction', 'how to', 'what is']
        }
        
        # Common precious metals
        self.metals = ['gold', 'silver', 'platinum', 'palladium', 'rhodium', 'iridium', 'ruthenium', 'osmium']
        
        # Conversation history (in-memory for demo, should use database in production)
        self.conversation_history = {}
    
    def ask_question(self, question: str, user_id: Optional[str] = None, context: Optional[Dict] = None) -> Dict:
        """
        Process a user question about precious metals
        
        Args:
            question: User's question
            user_id: Optional user identifier for conversation history
            context: Optional context (current prices, user preferences, etc.)
            
        Returns:
            Dict containing comprehensive answer with sources and analysis
        """
        try:
            # Analyze question to determine approach
            question_analysis = self._analyze_question(question)
            
            # Get real-time information if needed
            real_time_data = None
            if question_analysis['needs_current_data']:
                real_time_data = self._get_real_time_context(question, question_analysis)
            
            # Generate comprehensive answer
            answer_data = self._generate_comprehensive_answer(
                question, 
                question_analysis, 
                real_time_data, 
                context,
                user_id
            )
            
            # Store in conversation history
            if user_id:
                self._update_conversation_history(user_id, question, answer_data)
            
            return {
                'success': True,
                'question': question,
                'answer': answer_data,
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                'success': False,
                'question': question,
                'error': 'Failed to process question',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _analyze_question(self, question: str) -> Dict:
        """Analyze the question to determine the best approach"""
        question_lower = question.lower()
        
        # Determine category
        detected_categories = []
        for category, keywords in self.knowledge_categories.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_categories.append(category)
        
        # Detect mentioned metals
        mentioned_metals = [metal for metal in self.metals if metal in question_lower]
        
        # Determine if current data is needed
        current_data_keywords = ['current', 'today', 'now', 'latest', 'recent', 'price', 'market']
        needs_current_data = any(keyword in question_lower for keyword in current_data_keywords)
        
        # Determine complexity
        complexity_indicators = ['why', 'how', 'explain', 'compare', 'analyze', 'forecast', 'predict']
        is_complex = any(indicator in question_lower for indicator in complexity_indicators)
        
        return {
            'categories': detected_categories or ['general'],
            'metals': mentioned_metals,
            'needs_current_data': needs_current_data,
            'is_complex': is_complex,
            'question_type': self._classify_question_type(question_lower),
            'confidence': len(detected_categories) / len(self.knowledge_categories)
        }
    
    def _classify_question_type(self, question_lower: str) -> str:
        """Classify the type of question"""
        if any(word in question_lower for word in ['what', 'define', 'meaning']):
            return 'definition'
        elif any(word in question_lower for word in ['how', 'process', 'method']):
            return 'process'
        elif any(word in question_lower for word in ['why', 'reason', 'cause']):
            return 'explanation'
        elif any(word in question_lower for word in ['when', 'time', 'date']):
            return 'temporal'
        elif any(word in question_lower for word in ['where', 'location', 'place']):
            return 'location'
        elif any(word in question_lower for word in ['compare', 'difference', 'better']):
            return 'comparison'
        elif any(word in question_lower for word in ['should', 'recommend', 'advice']):
            return 'recommendation'
        elif any(word in question_lower for word in ['price', 'cost', 'value']):
            return 'pricing'
        else:
            return 'general'
    
    def _get_real_time_context(self, question: str, analysis: Dict) -> Dict:
        """Get real-time context using Perplexity AI"""
        try:
            # Construct context query
            metals_context = ', '.join(analysis['metals']) if analysis['metals'] else 'precious metals'
            context_query = f"Current market information about {metals_context}: {question}"
            
            headers = {
                'Authorization': f'Bearer {self.perplexity_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'sonar-pro',
                'messages': [
                    {
                        'role': 'user',
                        'content': f"""Provide current market context for this question: {context_query}
                        
                        Include:
                        1. Current prices and trends
                        2. Recent market developments
                        3. Relevant economic factors
                        4. Expert opinions or analysis
                        
                        Keep response factual and current."""
                    }
                ],
                'max_tokens': 800,
                'temperature': 0.2
            }
            
            response = requests.post(
                'https://api.perplexity.ai/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                context_content = result['choices'][0]['message']['content']
                
                return {
                    'real_time_context': context_content,
                    'source': 'perplexity_ai',
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                logger.warning(f"Perplexity API error for context: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting real-time context: {e}")
            return None
    
    def _generate_comprehensive_answer(self, question: str, analysis: Dict, 
                                     real_time_data: Optional[Dict], 
                                     context: Optional[Dict],
                                     user_id: Optional[str]) -> Dict:
        """Generate comprehensive answer using OpenAI"""
        try:
            if not self.openai_client:
                # Return demo answer when OpenAI is not available
                return {
                    'main_answer': f"I understand you're asking about: {question}\n\nWhile I'm currently in demo mode, I can tell you that this is a great question about precious metals. For the most accurate and up-to-date information, please ensure the AI services are properly configured.\n\nIn general, precious metals like gold, silver, platinum, and palladium are valuable investment assets that can serve as hedges against inflation and economic uncertainty. Each metal has unique properties and market dynamics that affect their prices and investment potential.",
                    'question_analysis': analysis,
                    'real_time_data_used': real_time_data is not None,
                    'follow_up_questions': [
                        "What specific aspect of precious metals interests you most?",
                        "Are you looking for investment advice or general information?",
                        "Would you like to know about current market conditions?"
                    ],
                    'related_topics': ["Investment strategies", "Market analysis", "Storage options", "Price factors"],
                    'confidence_score': 0.7,
                    'sources': ["PreciousAI Knowledge Base (Demo Mode)"],
                    'answer_type': analysis['question_type'],
                    'generated_at': datetime.utcnow().isoformat()
                }
            
            # Build context for OpenAI
            system_prompt = """You are MetAsk, an expert precious metals advisor with deep knowledge of:
            - Gold, silver, platinum, palladium, and other precious metals
            - Market analysis and investment strategies
            - Technical properties and industrial applications
            - Historical trends and future outlook
            - Storage, trading, and practical considerations
            
            Provide comprehensive, accurate, and helpful answers. Include:
            1. Direct answer to the question
            2. Supporting details and context
            3. Practical implications for investors
            4. Risk considerations when relevant
            5. Additional resources or next steps
            
            Be professional, informative, and accessible to both beginners and experts."""
            
            # Build user prompt with context
            user_prompt = f"Question: {question}\n\n"
            
            if real_time_data:
                user_prompt += f"Current Market Context:\n{real_time_data['real_time_context']}\n\n"
            
            if analysis['categories']:
                user_prompt += f"Question Categories: {', '.join(analysis['categories'])}\n"
            
            if analysis['metals']:
                user_prompt += f"Metals Mentioned: {', '.join(analysis['metals'])}\n"
            
            user_prompt += f"Question Type: {analysis['question_type']}\n"
            user_prompt += f"Complexity Level: {'High' if analysis['is_complex'] else 'Standard'}\n\n"
            
            user_prompt += "Please provide a comprehensive answer with practical insights."
            
            # Get conversation history for context
            conversation_context = ""
            if user_id and user_id in self.conversation_history:
                recent_history = self.conversation_history[user_id][-3:]  # Last 3 exchanges
                for item in recent_history:
                    conversation_context += f"Previous Q: {item['question']}\n"
                    conversation_context += f"Previous A: {item['answer']['main_answer'][:200]}...\n\n"
            
            if conversation_context:
                user_prompt += f"\n\nConversation Context:\n{conversation_context}"
            
            # Generate answer with OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1200,
                temperature=0.3
            )
            
            main_answer = response.choices[0].message.content
            
            # Generate follow-up questions
            follow_up_questions = self._generate_follow_up_questions(question, analysis, main_answer)
            
            # Generate related topics
            related_topics = self._generate_related_topics(analysis)
            
            return {
                'main_answer': main_answer,
                'question_analysis': analysis,
                'real_time_data_used': real_time_data is not None,
                'follow_up_questions': follow_up_questions,
                'related_topics': related_topics,
                'confidence_score': min(0.9, 0.6 + analysis['confidence']),
                'sources': self._get_answer_sources(real_time_data),
                'answer_type': analysis['question_type'],
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive answer: {e}")
            return {
                'main_answer': "I apologize, but I'm having trouble processing your question right now. Please try rephrasing your question or ask about a specific aspect of precious metals.",
                'error': str(e),
                'generated_at': datetime.utcnow().isoformat()
            }
    
    def _generate_follow_up_questions(self, original_question: str, analysis: Dict, answer: str) -> List[str]:
        """Generate relevant follow-up questions"""
        try:
            if not self.openai_client:
                # Return default follow-up questions
                return [
                    "What are the current market trends for this metal?",
                    "How does this compare to other precious metals?",
                    "What should investors consider for the future?"
                ]
            
            prompt = f"""Based on this precious metals question and answer, suggest 3 relevant follow-up questions:

            Original Question: {original_question}
            Answer Summary: {answer[:300]}...
            
            Generate questions that would naturally follow from this discussion."""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.5
            )
            
            follow_ups_text = response.choices[0].message.content
            
            # Extract questions from response
            questions = []
            for line in follow_ups_text.split('\n'):
                line = line.strip()
                if line and ('?' in line or line.startswith(('1.', '2.', '3.', '-', '•'))):
                    # Clean up the question
                    question = re.sub(r'^[\d\.\-\•\s]+', '', line).strip()
                    if question and len(question) > 10:
                        questions.append(question)
            
            return questions[:3]
            
        except Exception as e:
            logger.warning(f"Error generating follow-up questions: {e}")
            return [
                "What are the current market trends for this metal?",
                "How does this compare to other precious metals?",
                "What should investors consider for the future?"
            ]
    
    def _generate_related_topics(self, analysis: Dict) -> List[str]:
        """Generate related topics based on question analysis"""
        base_topics = [
            "Investment strategies",
            "Market analysis",
            "Storage and security",
            "Tax implications",
            "Industrial applications"
        ]
        
        category_topics = {
            'pricing': ["Price forecasting", "Market volatility", "Price drivers"],
            'investment': ["Portfolio allocation", "Risk management", "Investment vehicles"],
            'technical': ["Metal properties", "Purity standards", "Testing methods"],
            'market': ["Market trends", "Economic indicators", "Supply and demand"],
            'storage': ["Storage options", "Insurance", "Security measures"],
            'trading': ["Trading platforms", "Market makers", "Transaction costs"],
            'history': ["Historical performance", "Market cycles", "Long-term trends"],
            'industrial': ["Industrial demand", "Technology applications", "Future uses"]
        }
        
        related = base_topics.copy()
        for category in analysis.get('categories', []):
            if category in category_topics:
                related.extend(category_topics[category])
        
        return list(set(related))[:8]  # Return unique topics, max 8
    
    def _get_answer_sources(self, real_time_data: Optional[Dict]) -> List[str]:
        """Get sources used in the answer"""
        sources = ["PreciousAI Knowledge Base", "OpenAI GPT-4 Analysis"]
        
        if real_time_data:
            sources.append("Perplexity AI Real-time Data")
        
        return sources
    
    def _update_conversation_history(self, user_id: str, question: str, answer_data: Dict):
        """Update conversation history for user"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append({
            'question': question,
            'answer': answer_data,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Keep only last 10 exchanges per user
        if len(self.conversation_history[user_id]) > 10:
            self.conversation_history[user_id] = self.conversation_history[user_id][-10:]
    
    def get_conversation_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get conversation history for a user"""
        if user_id not in self.conversation_history:
            return []
        
        return self.conversation_history[user_id][-limit:]
    
    def get_popular_questions(self) -> List[Dict]:
        """Get popular/common questions about precious metals"""
        return [
            {
                'question': "What's the difference between gold and silver as investments?",
                'category': 'investment',
                'complexity': 'beginner'
            },
            {
                'question': "How do I store precious metals safely?",
                'category': 'storage',
                'complexity': 'beginner'
            },
            {
                'question': "What factors affect precious metals prices?",
                'category': 'pricing',
                'complexity': 'intermediate'
            },
            {
                'question': "Should I buy physical metals or ETFs?",
                'category': 'investment',
                'complexity': 'intermediate'
            },
            {
                'question': "How much of my portfolio should be in precious metals?",
                'category': 'investment',
                'complexity': 'intermediate'
            },
            {
                'question': "What is the best time to buy precious metals?",
                'category': 'market',
                'complexity': 'advanced'
            },
            {
                'question': "How do I verify the authenticity of precious metals?",
                'category': 'technical',
                'complexity': 'intermediate'
            },
            {
                'question': "What are the tax implications of precious metals investments?",
                'category': 'investment',
                'complexity': 'advanced'
            }
        ]
    
    def suggest_questions(self, topic: str) -> List[str]:
        """Suggest questions based on a topic"""
        topic_questions = {
            'gold': [
                "Why is gold considered a safe haven asset?",
                "How does gold perform during inflation?",
                "What's the difference between gold coins and bars?"
            ],
            'silver': [
                "Is silver a good investment for beginners?",
                "How does industrial demand affect silver prices?",
                "What's the gold-to-silver ratio and why does it matter?"
            ],
            'platinum': [
                "Why is platinum more expensive than gold?",
                "How does automotive demand affect platinum prices?",
                "Is platinum a good long-term investment?"
            ],
            'investment': [
                "How do I start investing in precious metals?",
                "What percentage of my portfolio should be precious metals?",
                "Physical metals vs. ETFs: which is better?"
            ],
            'market': [
                "What drives precious metals prices?",
                "How do economic conditions affect precious metals?",
                "What are the current market trends?"
            ]
        }
        
        return topic_questions.get(topic.lower(), [
            "What should I know about precious metals?",
            "How do I get started with precious metals investing?",
            "What are the current market conditions?"
        ])

