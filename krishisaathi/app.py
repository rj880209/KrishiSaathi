"""
Main Streamlit Application for KrishiSaathi
AI-powered agricultural assistant for Indian farmers
"""

import streamlit as st
import os
import json
from datetime import datetime
from PIL import Image
import tempfile

# Page configuration - must be first
st.set_page_config(
    page_title="KrishiSaathi - AI Agricultural Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .confidence-low {
        color: #D32F2F;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_retriever():
    """Load RAG retriever (cached)"""
    from utils.rag_engine import KnowledgeRetriever
    return KnowledgeRetriever("data/knowledge_base.json")


@st.cache_resource
def load_voice_processor():
    """Load voice processor (cached)"""
    from utils.voice_processor import VoiceProcessor
    return VoiceProcessor(groq_client=None)


@st.cache_resource
def load_classifier():
    """Load image classifier (cached)"""
    from utils.image_classifier import PestDiseaseClassifier
    return PestDiseaseClassifier()


@st.cache_resource
def load_escalation_handler():
    """Load escalation handler (cached)"""
    from utils.chat_engine import HumanEscalationHandler
    return HumanEscalationHandler()


@st.cache_resource
def load_offline_cache():
    """Load offline cache (cached)"""
    from utils.rag_engine import OfflineCache
    return OfflineCache()


def get_chat_engine(api_key):
    """Get chat engine (not cached - depends on API key)"""
    if not api_key:
        return None
    try:
        from utils.chat_engine import AIChatEngine
        return AIChatEngine(api_key=api_key)
    except Exception as e:
        st.warning(f"⚠️ Invalid API key. Using demo mode. Error: {str(e)}")
        return None


def initialize_session_state():
    """Initialize session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = "hi"
    if 'user_location' not in st.session_state:
        st.session_state.user_location = {"state": "", "district": ""}
    if 'soil_health_data' not in st.session_state:
        st.session_state.soil_health_data = None
    if 'message_counter' not in st.session_state:
        st.session_state.message_counter = 0
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = None
    if 'api_key_submitted' not in st.session_state:
        st.session_state.api_key_submitted = False


def render_sidebar(components):
    """Render sidebar with settings and info"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/agriculture.png", width=80)
        st.title("KrishiSaathi 🌾")
        st.markdown("**Your AI Farming Companion**")
        
        st.divider()
        
        # API Key Configuration
        st.subheader("🔑 API Configuration")
        
        # Show current status
        if st.session_state.groq_api_key:
            st.success("✅ Groq API Key configured")
            key_preview = "•" * 12 + st.session_state.groq_api_key[-4:] if len(st.session_state.groq_api_key) > 4 else "•" * 16
            st.markdown(f"**Key:** `{key_preview}`")
            
            if st.button("🔄 Change API Key", use_container_width=True):
                st.session_state.api_key_submitted = False
                st.session_state.groq_api_key = None
                st.rerun()
        else:
            api_key_input = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="gsk_...",
                help="Get your free API key from https://console.groq.com/keys",
                key="api_key_input_widget"
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("💾 Save Key", use_container_width=True, key="save_key_btn"):
                    if api_key_input.strip():
                        st.session_state.groq_api_key = api_key_input.strip()
                        st.session_state.api_key_submitted = True
                        st.success("✅ API Key saved!")
                        # Clear the input widget cache
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error("Please enter a valid API key")
            
            with col2:
                if st.button("🗑️ Clear", use_container_width=True, key="clear_key_btn", disabled=not api_key_input):
                    st.session_state.groq_api_key = None
                    st.session_state.api_key_submitted = False
                    st.rerun()
        
        st.divider()
        
        # Language selection
        st.subheader("🗣️ Language / भाषा")
        language_options = components['voice_processor'].get_language_options()
        selected_lang = st.selectbox(
            "Select Language",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=list(language_options.keys()).index(st.session_state.selected_language)
        )
        st.session_state.selected_language = selected_lang
        
        st.divider()
        
        # Location input
        st.subheader("📍 Your Location")
        col1, col2 = st.columns(2)
        with col1:
            state = st.text_input("State", value=st.session_state.user_location.get("state", ""))
        with col2:
            district = st.text_input("District", value=st.session_state.user_location.get("district", ""))
        
        if state and district:
            st.session_state.user_location = {"state": state, "district": district}
            # Preload FAQs for district
            if st.button("Download Offline FAQs"):
                components['offline_cache'].preload_top_100_faqs(
                    district, 
                    components['retriever']
                )
                st.success(f"✅ Cached FAQs for {district}")
        
        st.divider()
        
        # Soil Health Card integration
        st.subheader("📋 Soil Health Card")
        shc_id = st.text_input("Enter SHC Number", placeholder="e.g., MH12345678")
        if shc_id and st.button("Fetch Soil Data"):
            # Mock soil health data (integrate with actual API in production)
            st.session_state.soil_health_data = {
                'shc_id': shc_id,
                'npk': {'N': 'Low', 'P': 'Medium', 'K': 'High'},
                'ph': 6.8,
                'organic_carbon': '0.75%',
                'recommendations': 'Apply Nitrogen fertilizer. Phosphorus adequate.'
            }
            st.success("Soil Health Card data loaded!")
        
        if st.session_state.soil_health_data:
            st.json(st.session_state.soil_health_data)
        
        st.divider()
        
        # Quick stats
        st.subheader("📊 Session Stats")
        st.metric("Messages", len(st.session_state.conversation_history) // 2)
        
        # Show AI status
        if components['chat_engine']:
            st.success("🤖 AI Mode: Active")
        else:
            st.warning("🤖 Demo Mode")
        
        st.divider()
        
        # Talk to agent button
        st.subheader("👨‍🌾 Need Human Help?")
        if st.button("Talk to Agriculture Expert", use_container_width=True):
            st.info("Connecting you to an expert... (This would integrate with call center in production)")
            st.markdown("""
            **Expert Helpline Numbers:**
            - Kisan Call Center: 1800-180-1551
            - WhatsApp: +91-XXXXXXXXXX
            """)


def render_chat_interface(components):
    """Render main chat interface"""
    # Header
    st.markdown('<h1 class="main-header">🌾 KrishiSaathi</h1>', unsafe_allow_html=True)
    st.markdown("*AI-Powered Agricultural Assistant for Indian Farmers*")
    
    # Feature highlights
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="feature-box">🎤 Voice Input<br>5 Languages</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-box">📸 Pest Detection<br>Image Analysis</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-box">📊 Soil Health<br>Personalized Advice</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="feature-box">💬 24/7 Support<br>AI + Human</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Chat messages display
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.conversation_history):
            role = "user" if i % 2 == 0 else "assistant"
            with st.chat_message(role):
                st.write(message['content'])
                
                # Show confidence for assistant messages
                if role == "assistant" and 'confidence' in message:
                    conf = message['confidence']
                    conf_class = "confidence-high" if conf >= 0.85 else "confidence-low"
                    st.markdown(f'<span class="{conf_class}">Confidence: {conf:.0%}</span>', 
                              unsafe_allow_html=True)
                
                # Add feedback buttons for assistant messages
                if role == "assistant" and 'message_id' in message:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 Helpful", key=f"up_{message['message_id']}"):
                            components['chat_engine'].get_feedback(message['message_id'], 'thumbs_up')
                            st.success("Thank you for the feedback!")
                    with col2:
                        if st.button("👎 Not Helpful", key=f"down_{message['message_id']}"):
                            components['chat_engine'].get_feedback(message['message_id'], 'thumbs_down')
                            st.info("We'll improve our responses based on your feedback.")
    
    # Chat input
    st.divider()
    
    # Input methods tabs
    tab1, tab2, tab3 = st.tabs(["💬 Text Chat", "🎤 Voice Input", "📸 Image Upload"])
    
    with tab1:
        user_input = st.chat_input("Ask about crops, pests, prices, insurance...")
        if user_input:
            handle_user_message(user_input, components, source='text')
    
    with tab2:
        st.markdown("**Speak in your language:**")
        audio_file = st.file_uploader("Upload audio file (WAV format)", type=['wav', 'mp3'])
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_path = tmp_file.name
            
            with st.spinner("Transcribing..."):
                transcribed_text = components['voice_processor'].transcribe_audio(
                    tmp_path, 
                    st.session_state.selected_language
                )
                
                if transcribed_text:
                    st.success(f"🎤 You said: {transcribed_text}")
                    handle_user_message(transcribed_text, components, source='voice')
                else:
                    st.error("Could not transcribe audio. Please try again or use text input.")
                
                os.unlink(tmp_path)
    
    with tab3:
        st.markdown("**Upload crop/pest image for diagnosis:**")
        uploaded_image = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png'])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    predictions = components['classifier'].predict_from_pil(image, top_k=3)
                    
                    if predictions[0][0] != 'error':
                        st.subheader("🔍 Diagnosis Results:")
                        for disease, confidence in predictions:
                            emoji = "✅" if disease == 'healthy' else "⚠️"
                            st.write(f"{emoji} **{disease.replace('_', ' ').title()}**: {confidence:.1%}")
                        
                        # Show remediation for top prediction
                        top_disease = predictions[0][0]
                        if top_disease != 'healthy':
                            remediation = components['classifier'].get_remediation(top_disease)
                            st.info(f"""
                            **Recommended Treatment:**
                            
                            💊 Chemical: {remediation['chemical']}
                            
                            🌿 Organic: {remediation['organic']}
                            
                            📚 Source: {remediation['source']}
                            """)
                    else:
                        st.error("Could not analyze image. Please ensure it's a clear photo of the affected crop.")


def handle_user_message(user_input: str, components: dict, source: str = 'text'):
    """Process user message and generate response"""
    # Add user message to history
    st.session_state.conversation_history.append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().isoformat(),
        'source': source
    })
    
    # Get context from RAG
    context = components['retriever'].get_context_for_rag(user_input)
    
    # Generate response
    if components['chat_engine']:
        response, confidence, needs_escalation = components['chat_engine'].chat(
            user_input, 
            context=context,
            language=st.session_state.selected_language
        )
        
        # Handle low confidence - escalate to human
        if needs_escalation:
            queue_pos = components['escalation_handler'].escalate(
                user_input,
                response,
                confidence,
                user_contact=""  # Would get from user profile
            )
            response += f"\n\n⚠️ I'm not fully confident about this answer. Your query has been escalated to a human expert (Queue position: #{queue_pos}). They will contact you soon."
        
        # Add assistant response to history
        st.session_state.message_counter += 1
        st.session_state.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'message_id': f"msg_{st.session_state.message_counter}"
        })
        
        # Rerun to update UI
        st.rerun()
        
    else:
        # Demo mode response
        demo_response = f"""
        **Demo Mode Response:**
        
        You asked: "{user_input}"
        
        In production, this would be answered by our AI using:
        - RAG retrieval from verified knowledge base
        - Groq LLM for natural language generation
        - Context-aware follow-up suggestions
        
        Current context retrieved:
        {context[:500]}...
        """
        
        st.session_state.conversation_history.append({
            'role': 'assistant',
            'content': demo_response,
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.5,
            'message_id': f"msg_{st.session_state.message_counter}"
        })
        
        st.rerun()


def render_advanced_features(components):
    """Render advanced features section"""
    with st.expander("🚀 Advanced Features"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌦️ Predictive Alerts")
            st.markdown("""
            **Weather-Based Recommendations:**
            - Based on rainfall forecast, delay sowing by 3 days
            - High humidity alert: Fungal disease risk increased
            - Temperature warning: Protect livestock from heat stress
            """)
            
            st.subheader("💰 Market Price Coach")
            st.markdown("""
            **Real-time Price Intelligence:**
            - Buyers in Nashik paying ₹2,200/quintal for onions—hold your stock
            - Tomato prices expected to rise 15% next week in Pune APMC
            - Best time to sell: Thursday morning (higher demand)
            """)
        
        with col2:
            st.subheader("👥 FPO Group Chat")
            st.markdown("""
            **Farmer Producer Organization Features:**
            - Broadcast verified advisories to members
            - Group voice calls for training sessions
            - Shared market intelligence
            """)
            
            st.subheader("🏦 Credit Pre-Qualification")
            st.markdown("""
            **Account Aggregator Integration:**
            - Consent-based financial data sharing
            - Instant loan pre-approval
            - KCC renewal assistance
            """)


def main():
    """Main application"""
    initialize_session_state()
    
    # Load components using cached loaders for speed
    retriever = load_retriever()
    voice_processor = load_voice_processor()
    classifier = load_classifier()
    escalation_handler = load_escalation_handler()
    offline_cache = load_offline_cache()
    
    # Load chat engine (depends on API key, not cached)
    chat_engine = get_chat_engine(st.session_state.groq_api_key)
    
    if not chat_engine:
        st.info("🔑 Enter your Groq API Key in the sidebar to enable full AI features. Running in Demo Mode.")
    
    components = {
        'retriever': retriever,
        'chat_engine': chat_engine,
        'voice_processor': voice_processor,
        'classifier': classifier,
        'escalation_handler': escalation_handler,
        'offline_cache': offline_cache
    }
    
    # Render sidebar
    render_sidebar(components)
    
    # Render main chat interface
    render_chat_interface(components)
    
    # Render advanced features
    render_advanced_features(components)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>KrishiSaathi © 2024 | Powered by AI for Indian Farmers</p>
        <p>Data Sources: ICAR, Ministry of Agriculture, e-NAM, PMFBY</p>
        <p><small>For emergencies, contact Kisan Call Center: 1800-180-1551</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
