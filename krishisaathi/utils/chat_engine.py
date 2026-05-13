"""
AI Chat Engine with Groq integration
Handles LLM-powered responses with safety guards and confidence thresholds
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Try to import groq, provide mock mode if not available
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: groq package not installed. Running in demo mode.")
    print("Install with: pip install groq")

load_dotenv()

# Configure logging - Console only (no disk logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AIChatEngine:
    """Groq-powered chat engine with safety guards"""
    
    CONFIDENCE_THRESHOLD = 0.85
    
    def __init__(self, api_key: Optional[str] = None):
        if not GROQ_AVAILABLE:
            self.client = None
            self.model = "demo-mode"
            print("Running in demo mode - LLM responses disabled")
            return
            
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Please provide it via UI or .env file")
        
        self.client = Groq(api_key=self.api_key)
        self.conversation_history = []
        self.model = "llama-3.1-70b-versatile"
        
    def _build_system_prompt(self, context: str = "") -> str:
        """Build system prompt with agricultural expertise"""
        return f"""You are KrishiSaathi, an AI assistant for Indian farmers. You provide accurate, verified agricultural advice in simple language.

IMPORTANT GUIDELINES:
1. Only provide advice based on verified government/ICAR sources
2. If you're not certain about an answer (confidence < 85%), say "I recommend consulting a local agriculture expert for this specific question"
3. Use simple Hindi/English mix when appropriate
4. Always mention the source of your information
5. For pest/disease issues, suggest both organic and chemical remedies
6. Include cost estimates when relevant
7. Be empathetic to farmer concerns

Context from knowledge base:
{context}

Respond in a helpful, clear manner. If the question is outside your expertise, suggest talking to a human expert."""

    def chat(self, user_query: str, context: str = "", 
             language: str = "en") -> Tuple[str, float, bool]:
        """
        Process user query and return response
        
        Returns:
            tuple: (response_text, confidence_score, needs_human_escalation)
        """
        # Demo mode fallback
        if not GROQ_AVAILABLE or not self.client:
            return self._demo_response(user_query, context)
        
        try:
            # Add context-aware follow-up detection
            follow_up_suggestion = self._generate_follow_up(user_query)
            
            messages = [
                {"role": "system", "content": self._build_system_prompt(context)},
                {"role": "user", "content": user_query}
            ]
            
            # Add conversation history for context
            messages.extend(self.conversation_history[-5:])  # Last 5 messages
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=1024
            )
            
            assistant_message = response.choices[0].message.content
            confidence = self._estimate_confidence(assistant_message, context)
            
            # Log the interaction
            self._log_interaction(user_query, assistant_message, confidence, context)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_query})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Check if human escalation needed
            needs_escalation = confidence < self.CONFIDENCE_THRESHOLD
            
            # Add follow-up suggestion
            if follow_up_suggestion and not needs_escalation:
                assistant_message += f"\n\n💡 {follow_up_suggestion}"
            
            return assistant_message, confidence, needs_escalation
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return "I'm experiencing technical difficulties. Please try again or talk to a human expert.", 0.0, True
    
    def _demo_response(self, user_query: str, context: str) -> Tuple[str, float, bool]:
        """Generate demo response when Groq is not available"""
        # Use RAG context if available
        if context and len(context) > 50:
            response = f"""Based on our knowledge base:

{context[:500]}

**Demo Mode**: This is a sample response. In production, this would be generated by Groq LLM with full conversational capabilities.

💡 Would you like to know more about this topic?"""
            return response, 0.75, False
        
        return f"""You asked: "{user_query}"

**Demo Mode**: Install groq package and set GROQ_API_KEY for full AI functionality.

In production, I would provide:
- Verified agricultural advice from ICAR sources
- Context-aware follow-up questions  
- Multilingual support in 5 regional languages
- Integration with government APIs (Soil Health Card, MSP, PMFBY)""", 0.5, True
    
    def _estimate_confidence(self, response: str, context: str) -> float:
        """Estimate confidence score based on response quality and context match"""
        confidence = 0.7  # Base confidence
        
        # Increase confidence if response cites sources
        if any(keyword in response.lower() for keyword in ['source', 'icar', 'government', 'ministry']):
            confidence += 0.15
        
        # Increase confidence if context was used
        if context and len(context) > 50:
            confidence += 0.1
        
        # Decrease confidence if response indicates uncertainty
        uncertainty_phrases = ['i think', 'maybe', 'probably', 'not sure', 'consult an expert']
        if any(phrase in response.lower() for phrase in uncertainty_phrases):
            confidence -= 0.2
        
        return min(confidence, 1.0)
    
    def _generate_follow_up(self, query: str) -> Optional[str]:
        """Generate context-aware follow-up questions"""
        query_lower = query.lower()
        
        if 'wheat' in query_lower and ('rust' in query_lower or 'disease' in query_lower):
            return "Would you like to know about fungicide options available in your block?"
        elif 'msp' in query_lower or 'price' in query_lower:
            return "Would you like me to check current prices in your nearest APMC mandi?"
        elif 'soil' in query_lower or 'npk' in query_lower:
            return "Would you like personalized fertilizer recommendations based on your Soil Health Card?"
        elif 'pest' in query_lower or 'insect' in query_lower:
            return "Would you like to upload a photo for accurate pest identification?"
        elif 'insurance' in query_lower or 'pmfby' in query_lower:
            return "Would you like help with filing a claim or checking your policy status?"
        
        return None
    
    def _log_interaction(self, query: str, response: str, confidence: float, context: str):
        """Log interaction for audit and training purposes (console only)"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'confidence': confidence,
            'context_used': bool(context),
            'model': self.model
        }
        
        # Log to console only (no disk writing)
        logger.info(f"AUDIT: {json.dumps(log_entry, ensure_ascii=False)}")
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.info("Conversation history reset")
    
    def get_feedback(self, message_id: str, feedback: str):
        """
        Record user feedback for model improvement (console only)
        
        Args:
            message_id: Unique identifier for the message
            feedback: 'thumbs_up' or 'thumbs_down'
        """
        feedback_log = {
            'timestamp': datetime.now().isoformat(),
            'message_id': message_id,
            'feedback': feedback
        }
        
        # Log to console only (no disk writing)
        logger.info(f"FEEDBACK: {json.dumps(feedback_log, ensure_ascii=False)}")


class HumanEscalationHandler:
    """Handle escalation to human experts"""
    
    def __init__(self):
        self.expert_queue = []
    
    def escalate(self, user_query: str, ai_response: str, confidence: float, 
                  user_contact: str = ""):
        """Add query to human expert queue (console logging only)"""
        escalation = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'ai_response': ai_response,
            'confidence': confidence,
            'user_contact': user_contact,
            'status': 'pending'
        }
        
        self.expert_queue.append(escalation)
        
        # Log to console only (no disk writing)
        logger.warning(f"ESCALATION: {json.dumps(escalation, ensure_ascii=False)}")
        return len(self.expert_queue)  # Return queue position
