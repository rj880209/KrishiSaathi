"""
Pest and Disease Image Classification Module
Uses fine-tuned MobileNetV2/ViT for crop disease detection
Note: Requires torch and torchvision. Install with: pip install torch torchvision
For environments without torch, use mock mode for testing
"""

import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Try to import torch, fall back to mock mode if not available
try:
    from PIL import Image
    import numpy as np
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch/torchvision not installed. Running in mock mode.")
    print("Install with: pip install torch torchvision")


class PestDiseaseClassifier:
    """Image-based pest and disease classification"""
    
    # Common crop diseases and pests
    DISEASE_CLASSES = [
        'healthy',
        'wheat_rust_leaf',
        'wheat_rust_stripe',
        'rice_blast',
        'rice_brown_spot',
        'cotton_bollworm',
        'cotton_aphid',
        'potato_late_blight',
        'potato_early_blight',
        'tomato_leaf_spot',
        'tomato_yellow_leaf_curl',
        'maize_fall_armyworm',
        'sugarcane_red_rot',
        'chickpea_wilt',
        'soybean_rust'
    ]
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the classifier
        
        Args:
            model_path: Path to pre-trained model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if not TORCH_AVAILABLE:
            self.device = 'mock'
            self.model = None
            self.transform = None
            print("Running in mock mode - image classification disabled")
            return
            
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = len(self.DISEASE_CLASSES)
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load MobileNetV2 model with custom classifier head"""
        # Load pretrained MobileNetV2
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Replace classifier for our number of classes
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, self.num_classes)
        
        # Load custom weights if provided
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")
        else:
            print("Using randomly initialized weights (train the model for production use)")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self):
        """Get image preprocessing transforms"""
        if not TORCH_AVAILABLE:
            return None
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict disease/pest from image
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
        
        Returns:
            List of (class_name, confidence) tuples
        """
        if not TORCH_AVAILABLE:
            return [('mock_healthy', 0.95), ('mock_disease', 0.03)]
            
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, k=top_k)
            
            results = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                class_name = self.DISEASE_CLASSES[idx.item()]
                confidence = prob.item()
                results.append((class_name, confidence))
            
            return results
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return [('error', 0.0)]
    
    def predict_from_pil(self, image: Image.Image, top_k: int = 3) -> List[Tuple[str, float]]:
        """Predict from PIL Image object"""
        if not TORCH_AVAILABLE:
            return [('mock_healthy', 0.95), ('mock_disease', 0.03)]
            
        try:
            image = image.convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            top_probs, top_indices = torch.topk(probabilities, k=top_k)
            
            results = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                class_name = self.DISEASE_CLASSES[idx.item()]
                confidence = prob.item()
                results.append((class_name, confidence))
            
            return results
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return [('error', 0.0)]
    
    def get_remediation(self, disease_class: str) -> Dict[str, str]:
        """Get ICAR-approved remediation for detected disease"""
        remediations = {
            'wheat_rust_leaf': {
                'chemical': 'Spray Propiconazole 25% EC @ 0.1% or Tebuconazole 25.9% EC @ 0.1%',
                'organic': 'Use resistant varieties like HD-3086, HD-3226. Remove infected debris.',
                'source': 'ICAR-IIWBR'
            },
            'wheat_rust_stripe': {
                'chemical': 'Spray Propiconazole 25% EC @ 0.1%',
                'organic': 'Crop rotation, remove alternate hosts',
                'source': 'ICAR-IIWBR'
            },
            'rice_blast': {
                'chemical': 'Spray Tricyclazole 75% WP @ 0.6g/liter or Isoprothiolane 40% EC @ 1ml/liter',
                'organic': 'Use resistant varieties, balanced fertilization',
                'source': 'ICAR-IIRR'
            },
            'cotton_bollworm': {
                'chemical': 'Spray Emamectin Benzoate 5% SG @ 0.4g/liter or install pheromone traps',
                'organic': 'Release Trichogramma wasps, use Bt cotton varieties',
                'source': 'ICAR-CICR'
            },
            'cotton_aphid': {
                'chemical': 'Spray Imidacloprid 17.8% SL @ 0.3ml/liter',
                'organic': 'Release ladybird beetles, spray neem oil',
                'source': 'ICAR-CICR'
            },
            'potato_late_blight': {
                'chemical': 'Spray Mancozeb 75% WP @ 2.5g/liter or Metalaxyl 8% + Mancozeb 64% WP',
                'organic': 'Ensure drainage, avoid overhead irrigation, use certified seed',
                'source': 'ICAR-CPRI'
            },
            'tomato_leaf_spot': {
                'chemical': 'Spray Chlorothalonil 75% WP @ 2g/liter',
                'organic': 'Crop rotation, remove infected leaves',
                'source': 'ICAR-IIVR'
            }
        }
        
        return remediations.get(disease_class, {
            'chemical': 'Consult local agriculture expert for specific treatment',
            'organic': 'Maintain field hygiene and crop rotation',
            'source': 'General Advisory'
        })


def train_sample_model(output_path: str = "models/pest_classifier.pth", epochs: int = 1):
    """
    Placeholder for model training
    In production, this would train on a labeled dataset of crop images
    """
    print("Training placeholder - implement with actual dataset for production")
    
    # Create dummy model and save
    model = PestDiseaseClassifier()
    
    # Save model state (this is just a placeholder)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.model.state_dict(), output_path)
    print(f"Saved placeholder model to {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Test the classifier
    print("Initializing Pest & Disease Classifier...")
    classifier = PestDiseaseClassifier()
    
    print(f"\nDevice: {classifier.device}")
    print(f"Number of classes: {classifier.num_classes}")
    print(f"Classes: {classifier.DISEASE_CLASSES[:5]}...")  # Show first 5
    
    # Test remediation lookup
    test_disease = 'wheat_rust_leaf'
    remediation = classifier.get_remediation(test_disease)
    print(f"\nRemediation for {test_disease}:")
    print(f"Chemical: {remediation['chemical']}")
    print(f"Organic: {remediation['organic']}")
    print(f"Source: {remediation['source']}")
