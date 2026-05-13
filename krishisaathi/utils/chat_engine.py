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
            # Don't raise error, just use demo mode
            self.client = None
            self.model = "demo-mode"
            print("No API key provided - running in demo mode with RAG responses")
            return
        
        try:
            self.client = Groq(api_key=self.api_key)
            self.conversation_history = []
            self.model = "llama-3.1-70b-versatile"
        except Exception as e:
            print(f"Invalid API key - running in demo mode. Error: {e}")
            self.client = None
            self.model = "demo-mode"
        
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
        if context and len(context) > 50 and "No specific information" not in context:
            # Extract the first answer from context
            lines = context.split('\n')
            answer_lines = []
            found_answer = False
            for line in lines:
                if line.startswith('Answer:'):
                    found_answer = True
                    answer_lines.append(line.replace('Answer:', '').strip())
                elif found_answer and (line.startswith('Source:') or line.startswith('---')):
                    break
                elif found_answer:
                    answer_lines.append(line.strip())
            
            if answer_lines:
                answer = ' '.join(answer_lines)
                response = f"""{answer}

**Source**: Verified government/ICAR advisory

💡 Would you like to know more about this topic?"""
                return response, 0.90, False
        
        # Fallback: try to match common queries
        query_lower = user_query.lower()
        
        # Common agriculture queries with pre-defined responses
        if any(word in query_lower for word in ['npk', 'nitrogen', 'phosphorus', 'potassium']):
            return """NPK stands for Nitrogen (N), Phosphorus (P), and Potassium (K) - the three primary nutrients essential for plant growth.

• **Nitrogen (N)**: Promotes leaf and stem growth
• **Phosphorus (P)**: Supports root development and flowering  
• **Potassium (K)**: Enhances overall plant health and disease resistance

Check your Soil Health Card for personalized NPK recommendations based on your soil test results.

**Source**: ICAR Soil Science Guidelines

💡 Would you like personalized fertilizer recommendations based on your Soil Health Card?""", 0.92, False
            
        elif any(word in query_lower for word in ['msp', 'minimum support price']):
            return """MSP (Minimum Support Price) is the price at which the government purchases crops from farmers.

**Current MSP for major crops (2024)**:
• Wheat: ₹2,275/quintal
• Rice (Common): ₹2,300/quintal  
• Maize: ₹2,090/quintal
• Cotton: ₹6,620/quintal

Prices are announced by the Government of India before each sowing season.

**Source**: Ministry of Agriculture & Farmers Welfare

💡 Would you like me to check current prices in your nearest APMC mandi?""", 0.91, False
            
        elif any(word in query_lower for word in ['wheat', 'गेहूं']) and any(word in query_lower for word in ['yellow', 'पीले', 'leaf', 'पत्ते']):
            return """Yellowing of wheat leaves can be caused by several factors:

**Common Causes**:
1. **Nitrogen deficiency** - Most common cause
2. **Waterlogging** - Poor drainage
3. **Rust disease** - Orange/brown pustules
4. **Zinc deficiency** - Interveinal chlorosis

**Recommended Actions**:
• Apply Urea @ 25-30 kg/acre if nitrogen deficient
• Ensure proper field drainage
• For rust: Spray Propiconazole 0.1%
• For zinc: Apply ZnSO4 @ 25 kg/acre

**Source**: ICAR-Indian Institute of Wheat & Barley Research

💡 Would you like to know about fungicide options available in your block?""", 0.91, False
        
        return f"""You asked: "{user_query}"

**Demo Mode**: This is a sample response. In production with a valid Groq API key, I would provide:
• Verified agricultural advice from ICAR sources
• Context-aware follow-up questions  
• Multilingual support in 5 regional languages
• Integration with government APIs (Soil Health Card, MSP, PMFBY)

For immediate assistance, contact Kisan Call Center: 1800-180-1551""", 0.65, True
    
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
