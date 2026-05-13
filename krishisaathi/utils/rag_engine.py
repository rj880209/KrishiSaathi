"""
RAG-based knowledge retrieval system for KrishiSaathi
Uses vector embeddings and semantic search for accurate answer retrieval
"""

import json
import os
from typing import List, Dict, Tuple
import numpy as np

# Try to import sentence_transformers, fall back to simple keyword matching
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence_transformers not installed. Using keyword-based retrieval.")
    print("Install with: pip install sentence-transformers")


class KnowledgeRetriever:
    """Retrieval-Augmented Generation system for agricultural knowledge"""
    
    def __init__(self, knowledge_base_path: str = "data/knowledge_base.json"):
        self.kb_path = knowledge_base_path
        self.documents = []
        self.embeddings = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model = None
            
        self.load_knowledge_base()
        
    def load_knowledge_base(self):
        """Load and embed the knowledge base"""
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        self.documents = []
        # Flatten all questions into documents
        for category, data in kb_data.items():
            for qa in data.get('questions', []):
                doc = {
                    'id': qa['id'],
                    'category': category,
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'source': qa['source'],
                    'tags': qa.get('tags', [])
                }
                self.documents.append(doc)
        
        # Create embeddings if sentence_transformers is available
        if self.documents and SENTENCE_TRANSFORMERS_AVAILABLE and self.model:
            texts = [f"{doc['question']} {doc['answer']}" for doc in self.documents]
            self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        elif self.documents:
            # Fallback: store text representations for keyword matching
            self.texts = [f"{doc['question']} {doc['answer']}".lower() for doc in self.documents]
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Retrieve most relevant documents for a query"""
        if not self.documents:
            return []
        
        # Use semantic search if embeddings are available
        if self.embeddings is not None and self.model is not None:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
            
            # Calculate cosine similarity
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Minimum similarity threshold
                    results.append((self.documents[idx], float(similarities[idx])))
            
            return results
        
        # Fallback: keyword-based retrieval
        else:
            query_lower = query.lower()
            scores = []
            
            for i, text in enumerate(self.texts):
                # Simple keyword overlap score
                query_words = set(query_lower.split())
                doc_words = set(text.split())
                overlap = len(query_words & doc_words) / max(len(query_words), 1)
                scores.append((i, overlap))
            
            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for idx, score in scores[:top_k]:
                if score > 0.1:  # Minimum threshold
                    results.append((self.documents[idx], float(score)))
            
            return results
    
    def get_context_for_rag(self, query: str) -> str:
        """Get formatted context for RAG-powered responses"""
        results = self.retrieve(query, top_k=3)
        
        if not results:
            return "No specific information found in knowledge base."
        
        context_parts = []
        for doc, score in results:
            context_parts.append(
                f"Source: {doc['source']}\n"
                f"Question: {doc['question']}\n"
                f"Answer: {doc['answer']}\n"
                f"Confidence: {score:.2f}"
            )
        
        return "\n\n---\n\n".join(context_parts)


class OfflineCache:
    """Offline caching system for district-specific FAQs"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def cache_district_faqs(self, district: str, faqs: List[Dict]):
        """Cache top FAQs for a district"""
        cache_file = os.path.join(self.cache_dir, f"{district.lower().replace(' ', '_')}_faqs.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(faqs, f, indent=2, ensure_ascii=False)
    
    def get_cached_faqs(self, district: str) -> List[Dict]:
        """Retrieve cached FAQs for a district"""
        cache_file = os.path.join(self.cache_dir, f"{district.lower().replace(' ', '_')}_faqs.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def preload_top_100_faqs(self, district: str, retriever: KnowledgeRetriever):
        """Preload top 100 FAQs for offline access"""
        # This would typically use historical query data for the district
        # For now, we'll cache the entire knowledge base
        faqs = retriever.documents[:100]  # Top 100 from KB
        self.cache_district_faqs(district, faqs)
