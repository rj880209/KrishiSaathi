# KrishiSaathi 🌾

AI-powered agricultural assistant for Indian farmers, providing multilingual voice support, pest diagnosis, and verified agricultural advice.

## Features

### Core Functionality
- **🎤 Voice Input in 5 Regional Languages**: Hindi, Telugu, Marathi, Bengali, Tamil
- **💬 Pre-defined Q&A** on:
  - Soil Health Card interpretation
  - MSP & e-NAM prices
  - PMFBY insurance claims
  - Common pest remedies (ICAR-approved)
- **📸 Image-based Pest/Disease Diagnosis** using fine-tuned MobileNetV2
- **🔍 RAG-powered Answers** from verified agricultural knowledge graphs
- **💡 Context-aware Follow-ups** for better user engagement

### Safety & Compliance
- **Confidence Threshold**: AI certainty <85% routes to human expert
- **Audit Logging**: Every advice logged with source + timestamp
- **Feedback Loop**: 👍/👎 ratings to retrain models
- **Verified Sources**: ICAR, Ministry of Agriculture, e-NAM, PMFBY

### Advanced Features
- **🌦️ Predictive Alerts**: Weather-based sowing/harvest recommendations
- **💰 Market Price Coach**: Real-time price intelligence and negotiation tips
- **👥 FPO Group Chats**: Broadcast verified advisories to farmer groups
- **🏦 Credit Pre-Qualification**: Account Aggregator integration for loans

## Tech Stack

- **Backend**: Python 3.10+
- **Frontend**: Streamlit
- **AI/LLM**: Groq (Llama 3.1 70B)
- **Speech-to-Text**: Groq Whisper / Google Speech Recognition (fallback)
- **NLP/RAG**: Sentence Transformers, LangChain
- **Image Classification**: PyTorch, MobileNetV2
- **Vector Store**: In-memory with NumPy (can scale to ChromaDB)

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Groq API key (get free key at https://console.groq.com/keys)

### Setup Steps

1. **Clone the repository**
```bash
cd krishisaathi
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

> **Note**: For minimal setup (text chat only), install just the core packages:
> ```bash
> pip install streamlit groq requests Pillow python-dotenv
> ```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Configure API Key in UI**
   - Open the sidebar (☰ menu)
   - Navigate to "🔑 API Configuration" section
   - Enter your Groq API key
   - Click "💾 Save Key"
   - The app will immediately switch to AI mode!

### Alternative: Environment Variable
You can also set the API key via environment variable:
```bash
export GROQ_API_KEY="your_api_key_here"
streamlit run app.py
```

**Note**: UI-provided API keys take precedence over environment variables and are stored securely in session state (cleared when browser closes).

2. **Create virtual environment** (optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables** (optional, can use UI instead)
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

5. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
krishisaathi/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── data/
│   ├── knowledge_base.json    # Structured Q&A database
│   └── cache/                 # Offline FAQ cache
├── utils/
│   ├── rag_engine.py          # RAG retrieval system
│   ├── chat_engine.py         # Groq LLM integration
│   ├── voice_processor.py     # Speech-to-text processing
│   └── image_classifier.py    # Pest/disease image analysis
├── models/                    # Trained ML models
└── logs/                      # Audit and feedback logs
```

## 🧪 Test Data & Validation Guide

Use the following test cases to verify all features of KrishiSaathi.

### 1. 🔑 API Configuration
| Action | Input | Expected Result |
| :--- | :--- | :--- |
| **Save Key** | Enter valid Groq Key (e.g., `gsk_...`) | Green success message, "AI Mode" activates |
| **Clear Key** | Click "Clear" | Key removed, app switches to "Demo Mode" |
| **Invalid Key** | Enter `fake_key_123` | Error toast on first AI query attempt |

### 2. 🗣️ Voice Input (Regional Languages)
*Select language from dropdown and speak (or upload audio file if supported)*

| Language | Test Phrase (Speak/Type) | Expected Intent |
| :--- | :--- | :--- |
| **Hindi** | "गेहूं के पत्ते पीले पड़ रहे हैं" | Pest/Disease Detection (Wheat) |
| **Hindi** | "मैं अपनी मिट्टी की जांच कैसे करूं?" | Soil Health Card Query |
| **Hindi** | "गेहूं का एमएसपी क्या है?" | MSP Price Query |
| **Telugu** | "వరి ధర ఎంత?" (Rice price?) | MSP/Market Price Query |
| **Telugu** | "నా పంట ఆరోగ్యం ఎలా తనిఖీ చేసుకోవాలి?" | Crop Health Check |
| **Marathi** | "कापूस विमा कसा काढावा?" | PMFBY Insurance Query |
| **Marathi** | "माझ्या पिकाची तब्येत कशी तपासावी?" | Crop Health Check |
| **Bengali** | "মাটির স্বাস্থ্য কার্ড কীভাবে পাব?" | Soil Health Card Query |
| **Bengali** | "ধানের ব্লাস্ট রোগ কীভাবে চিহ্নিত করব?" | Rice Blast Disease |
| **Tamil** | "நெல் பயிர் நோய் தீர்வு" | Pest Remedy Query |
| **Tamil** | "என் பயிரின் ஆரோக்கியத்தை எப்படி சரிபார்க்கலாம்?" | Crop Health Check |

### 3. 💬 Pre-defined Q&A (RAG System)
*Type these exact queries to test knowledge base retrieval.*

#### A. Soil Health Card
- "What does NPK stand for in my soil card?"
- "How to interpret NPK values in Soil Health Card?"
- "My soil has low Nitrogen, what fertilizer should I use?"
- "What is a Soil Health Card?"
- "How to interpret EC value in Soil Health Card?"
- "What does pH value indicate?"
- "Ideal pH for crops"

#### B. MSP & e-NAM Prices
- "What is the current MSP for Wheat?"
- "What is MSP for wheat in 2024?"
- "What is MSP for paddy/rice?"
- "Show me onion prices in Nashik e-NAM market."
- "How to sell on e-NAM?"
- "Is there a bonus for organic farming under MSP?"

#### C. PMFBY Insurance
- "What is PMFBY?"
- "How to claim insurance for crop loss due to hail?"
- "How to file a claim under PMFBY?"
- "What is the premium rate for Kharif crops?"
- "What is the premium rate under PMFBY?"
- "Last date for PMFBY enrollment?"

#### D. Pest Remedies (ICAR Approved)
- "Remedy for Brown Spot in Rice."
- "How to control Aphids in Mustard?"
- "Organic pesticide for Tomato blight."
- "How to control wheat rust?"
- "How to control pest in cotton?"
- "How to manage blight in potato?"
- "गेहूं की फसल में पत्ते पीले क्यों हो रहे हैं?" (Hindi: Why are wheat leaves turning yellow?)
- "wheat yellow leaves treatment"
- "cotton bollworm remedy"

### 4. 🖼️ Image Diagnosis (Pest/Disease)
*Upload sample images to test the classifier.*

| Crop | Disease/Pest | Expected Advice |
| :--- | :--- | :--- |
| **Tomato** | Early Blight (Dark spots with rings) | Fungicide recommendation + Cultural practices |
| **Wheat** | Rust (Orange powdery spores) | Immediate fungicide alert + Variety resistance info |
| **Rice** | Blast (Diamond shaped lesions) | Water management advice + Tricyclazole suggestion |
| **Cotton** | Bollworm (Holes in bolls) | Bt-cotton info + Pheromone trap suggestion |
| **Potato** | Late Blight (Water-soaked lesions) | Mancozeb spray + Drainage improvement |
| **Healthy** | Any healthy crop image | "No issues detected" message |

*(Note: In Demo Mode without model weights, this returns simulated diagnosis)*

### 5. 🛡️ Safety Guards & Confidence Threshold
*Test the fallback mechanism.*

| Test Case | Input Query | Expected Behavior |
| :--- | :--- | :--- |
| **Low Confidence** | "What is the best stock to buy today?" (Out of domain) | Confidence <85% → Trigger "Talk to Agent" button |
| **Ambiguous** | "It is not working." | Clarification question OR Route to Agent |
| **High Risk** | "Can I eat this pesticide?" | Immediate Warning + Human Expert Escalation |
| **Out of Scope** | "How to fix my car engine?" | Route to human expert with escalation message |
| **Vague Query** | "My plant is sick" | Request for more details or image upload |

### 6. 🔄 Context-Aware Follow-ups
*Test multi-turn conversation memory.*

**Scenario 1: Wheat Disease**
1. **User:** "Tell me about Wheat Rust." / "गेहूं में रस्ट रोग"
   - *System:* Explains Wheat Rust with treatment.
2. **User:** "What are the chemical options?"
   - *System:* Should understand context is still **Wheat Rust** and list fungicides (Propiconazole, Tebuconazole).
3. **User:** "Is it available in my block?"
   - *System:* Should ask for location/block details to check availability.

**Scenario 2: MSP Query**
1. **User:** "What is MSP for rice?"
   - *System:* Provides MSP rates for paddy.
2. **User:** "When is the best time to sell?"
   - *System:* Should provide mandi timing/market advice.
3. **User:** "How to register on e-NAM?"
   - *System:* Provides registration steps.

**Scenario 3: Soil Health**
1. **User:** "My soil test shows low nitrogen"
   - *System:* Recommends nitrogen fertilizers.
2. **User:** "How much urea should I apply?"
   - *System:* Provides dosage based on crop type.

### 7. 📝 Feedback Loop
| Action | Expected Result |
| :--- | :--- |
| Click 👍 on good response | "Thank you!" toast, positive log entry in console |
| Click 👎 on poor response | "Sorry! Improving..." toast, negative log entry for retraining |
| Skip feedback | No action, conversation continues |

### 8. 📡 Advanced Features (UI Mockups)
| Feature | Test Action | Expected UI Response |
| :--- | :--- | :--- |
| **Predictive Alerts** | View Dashboard | Weather-based sowing advice card visible |
| **Market Coach** | Select "Onion" | Price trend graph + "Hold/Sell" recommendation |
| **FPO Chat** | Type in Group Chat | Message broadcast simulation |
| **Credit Pre-qual** | Click "Check Eligibility" | Mock credit score display based on AA framework |
| **Offline Cache** | Enter District + Click Download | Success message, FAQs cached for offline use |
| **Soil Health Card** | Enter SHC number + Fetch | Mock NPK data displayed with recommendations |

---

## 🚀 Running Tests
1. **Start App:** `streamlit run app.py`
2. **Configure:** Enter Groq API Key in sidebar (optional for demo mode).
3. **Execute:** Go through sections 1-8 above systematically.
4. **Verify Console Logs:** Watch for RAG retrieval messages, confidence scores, and audit logs in console output.
5. **Test Edge Cases:** Try misspelled queries, mixed language inputs, and very short/long questions.

### Quick Smoke Test Commands
```bash
# Test RAG retrieval directly
python -c "from utils.rag_engine import KnowledgeRetriever; r = KnowledgeRetriever('data/knowledge_base.json'); print(r.retrieve('wheat rust', top_k=2))"

# Test chat engine in demo mode
python -c "from utils.chat_engine import AIChatEngine; c = AIChatEngine(); print(c.chat('What is MSP for wheat?', context='MSP is 2275'))"

# Verify knowledge base loading
python -c "import json; kb = json.load(open('data/knowledge_base.json')); print(f'Loaded {sum(len(v[\"questions\"]) for v in kb.values())} Q&A pairs')"
```

---

## Usage Guide

### Text Chat
1. Select your preferred language from the sidebar
2. Type your question in the chat input
3. Get instant AI-powered responses with confidence scores
4. Provide feedback with 👍/👎 buttons

### Voice Input
1. Select your language (Hindi, Telugu, Marathi, Bengali, Tamil)
2. Upload an audio file (WAV/MP3 format)
3. System transcribes and processes your query
4. Receive spoken-response in text format

### Pest Diagnosis
1. Navigate to "Image Upload" tab
2. Upload a clear photo of affected crop/pest
3. Get diagnosis with top 3 predictions
4. View ICAR-approved treatment recommendations

### Soil Health Integration
1. Enter your Soil Health Card number in sidebar
2. Click "Fetch Soil Data"
3. Get personalized NPK recommendations
4. Receive fertilizer suggestions based on soil test

## API Integration Points

### Government APIs (Production)
- **Soil Health Card**: `https://soilhealth.dac.gov.in/api`
- **PMFBY Insurance**: `https://pmfby.gov.in/api`
- **e-NAM Prices**: `https://enam.gov.in/api`
- **IMD Weather**: `https://mausam.imd.gov.in/api`

### AI Services
- **Groq LLM**: Llama 3.1 70B for natural language generation
- **Groq Speech**: Whisper-large-v3 for transcription
- **HuggingFace**: Sentence transformers for embeddings

## Safety & Ethics

### Confidence Scoring
- Responses include confidence scores (0-100%)
- <85% confidence triggers human expert escalation
- All low-confidence queries logged for review

### Data Privacy
- Farmer data stored locally (can integrate with secure cloud)
- No personal data sent to LLM without consent
- Audit trail for all interactions

### Verified Information
- Knowledge base sourced from ICAR, government ministries
- Regular updates with latest advisories
- Source attribution for every response

## Development

### Adding New Q&A
Edit `data/knowledge_base.json`:
```json
{
  "category_name": {
    "questions": [
      {
        "id": "Q001",
        "question": "Your question here?",
        "answer": "Verified answer here",
        "source": "Source organization",
        "tags": ["tag1", "tag2"]
      }
    ]
  }
}
```

### Training Image Classifier
```python
from utils.image_classifier import train_sample_model

# Train with your dataset
train_sample_model(
    output_path="models/custom_classifier.pth",
    epochs=10
)
```

### Custom Language Support
Add language to `utils/voice_processor.py`:
```python
class Language(Enum):
    YOUR_LANG = "xx"  # ISO language code
```

## Deployment

### Local Development
```bash
streamlit run app.py --server.port 8501
```

### Production (Streamlit Cloud)
1. Push code to GitHub
2. Connect repo to Streamlit Cloud
3. Set GROQ_API_KEY in secrets
4. Deploy

### Docker Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## Troubleshooting

### Common Issues

**GROQ_API_KEY Error**
- Ensure `.env` file exists with valid API key
- Check key has no extra spaces/quotes

**Model Loading Error**
- First run downloads sentence-transformers model (~90MB)
- Ensure stable internet connection
- Models cached in `~/.cache/torch/sentence_transformers`

**Audio Transcription Fails**
- Verify audio format (WAV preferred)
- Check sample rate (8kHz-48kHz supported)
- Try fallback Google Speech Recognition

**Image Classification Slow**
- Use CPU mode if no GPU available
- Reduce image resolution before upload
- Consider quantized model for production

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- **ICAR** (Indian Council of Agricultural Research) for technical guidelines
- **Ministry of Agriculture & Farmers Welfare** for data sources
- **Groq** for fast LLM inference
- **Streamlit** for amazing UI framework
- **HuggingFace** for open-source models

## Contact

- **Kisan Call Center**: 1800-180-1551 (Toll-free)
- **Email**: support@krishisaathi.in (placeholder)
- **Website**: www.krishisaathi.in (placeholder)

---

**Built with ❤️ for Indian Farmers**
