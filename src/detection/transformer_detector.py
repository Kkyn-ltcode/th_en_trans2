import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
from .base_detector import BaseCandidateDetector, TransliterationCandidate
from ..utils.tokenizer import ThaiTokenizer

class TransformerDetector(BaseCandidateDetector):
    """
    Transformer-based detector for transliteration candidates
    """
    def __init__(self, 
                 model_name_or_path: str = "airesearch/wangchanberta-base-att-spm-uncased",
                 tokenizer_name_or_path: Optional[str] = None,
                 thai_tokenizer: Optional[ThaiTokenizer] = None,
                 device: Optional[str] = None,
                 name: str = "transformer_detector"):
        """
        Initialize the transformer-based detector
        
        Args:
            model_name_or_path: Pretrained model name or path
            tokenizer_name_or_path: Tokenizer name or path (if different from model)
            thai_tokenizer: ThaiTokenizer instance for Thai word segmentation
            device: Device to run inference on ('cuda', 'cpu', etc.)
            name: Name of the detector
        """
        super().__init__(name=name)
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize Thai tokenizer
        self.thai_tokenizer = thai_tokenizer or ThaiTokenizer()
        
        # Load transformer tokenizer
        tokenizer_name = tokenizer_name_or_path or model_name_or_path
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Check if model exists and is a fine-tuned classifier
        if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
            # Load fine-tuned model
            self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
            self.is_trained = True
        else:
            # Load base model, will need fine-tuning
            self.logger.warning(
                f"Loading base model {model_name_or_path}. "
                "This model needs to be fine-tuned for transliteration detection."
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name_or_path, 
                num_labels=3  # B-TRANS, I-TRANS, O
            )
            self.is_trained = False
            
        # Move model to device
        self.model.to(self.device)
        
        # Label mapping
        self.id2label = {0: "O", 1: "B-TRANS", 2: "I-TRANS"}
        self.label2id = {"O": 0, "B-TRANS": 1, "I-TRANS": 2}
        
    def fine_tune(self, 
                 train_texts: List[str], 
                 train_labels: List[List[str]],
                 val_texts: Optional[List[str]] = None,
                 val_labels: Optional[List[List[str]]] = None,
                 epochs: int = 3,
                 batch_size: int = 8,
                 learning_rate: float = 5e-5,
                 save_path: Optional[str] = None):
        """
        Fine-tune the transformer model for transliteration detection
        
        Args:
            train_texts: List of training texts
            train_labels: List of BIO labels for each token in training texts
            val_texts: List of validation texts (optional)
            val_labels: List of BIO labels for validation texts (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            save_path: Path to save the fine-tuned model
        """
        # This is a placeholder for the fine-tuning code
        # In a real implementation, you would:
        # 1. Prepare datasets (convert texts and labels to tensors)
        # 2. Set up training arguments
        # 3. Use Trainer from transformers to fine-tune
        # 4. Save the model
        
        self.logger.info("Fine-tuning not implemented in this example")
        self.logger.info("In a real implementation, you would use HuggingFace Trainer")
        
        # Mark as trained
        self.is_trained = True
        
        # Save model if path provided
        if save_path:
            self.model.save_pretrained(save_path)
            self.transformer_tokenizer.save_pretrained(save_path)
            self.logger.info(f"Model saved to {save_path}")
    
    def detect_candidates(self, text: str) -> List[TransliterationCandidate]:
        """
        Detect transliteration candidates using the transformer model
        
        Args:
            text: Thai text to analyze
            
        Returns:
            List of transliteration candidates
        """
        if not self.is_trained:
            self.logger.error("Model not fine-tuned. Fine-tune the model first.")
            return []
        
        candidates = []
        
        # First tokenize with Thai tokenizer to get word boundaries
        thai_tokens = self.thai_tokenizer.tokenize(text)
        
        # Then tokenize with transformer tokenizer
        # This is a simplified approach - in practice, you'd need to handle
        # alignment between Thai tokens and transformer tokens carefully
        inputs = self.transformer_tokenizer(
            thai_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Get predicted labels
        predictions = torch.argmax(logits, dim=2)
        
        # Convert predictions to BIO tags
        predicted_labels = []
        for i in range(predictions.shape[1]):
            if i < len(thai_tokens):
                label_id = predictions[0, i].item()
                predicted_labels.append(self.id2label[label_id])
        
        # Extract transliteration candidates based on BIO tags
        current_candidate = None
        start_pos = 0
        
        for i, (token, label) in enumerate(zip(thai_tokens, predicted_labels)):
            # Update start position
            if i > 0:
                start_pos += len(thai_tokens[i-1])
                
            end_pos = start_pos + len(token)
            
            if label == "B-TRANS":
                # Start a new candidate
                current_candidate = {
                    "tokens": [token],
                    "start": start_pos,
                    "end": end_pos,
                    "confidence": self._get_confidence(logits[0, i])
                }
            elif label == "I-TRANS" and current_candidate is not None:
                # Continue current candidate
                current_candidate["tokens"].append(token)
                current_candidate["end"] = end_pos
                current_candidate["confidence"] = (
                    current_candidate["confidence"] + self._get_confidence(logits[0, i])
                ) / 2  # Average confidence
            elif current_candidate is not None:
                # End of candidate
                candidate_text = "".join(current_candidate["tokens"])
                candidates.append(
                    TransliterationCandidate(
                        token=candidate_text,
                        start_pos=current_candidate["start"],
                        end_pos=current_candidate["end"],
                        confidence=current_candidate["confidence"],
                        metadata={"detection_method": "transformer"}
                    )
                )
                current_candidate = None
        
        # Handle case where the last token is part of a candidate
        if current_candidate is not None:
            candidate_text = "".join(current_candidate["tokens"])
            candidates.append(
                TransliterationCandidate(
                    token=candidate_text,
                    start_pos=current_candidate["start"],
                    end_pos=current_candidate["end"],
                    confidence=current_candidate["confidence"],
                    metadata={"detection_method": "transformer"}
                )
            )
        
        return candidates
    
    def _get_confidence(self, logits: torch.Tensor) -> float:
        """
        Calculate confidence score from logits
        
        Args:
            logits: Logits from the model for a single token
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=0)
        
        # Get probability of B-TRANS or I-TRANS (whichever is higher)
        trans_prob = max(probs[1].item(), probs[2].item())
        
        return trans_prob