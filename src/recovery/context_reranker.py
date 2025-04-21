import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

class ContextReranker:
    """
    Reranks transliteration candidates based on context
    """
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize the context reranker
        
        Args:
            model_name: Name of the pretrained language model
            device: Device to run inference on ('cuda', 'cpu', etc.)
            cache_dir: Directory to cache models
        """
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Special tokens
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
    
    def rerank_candidates(self, 
                         candidates: List[str], 
                         context_before: str, 
                         context_after: str) -> List[Tuple[str, float]]:
        """
        Rerank candidates based on context
        
        Args:
            candidates: List of English word candidates
            context_before: Text context before the word
            context_after: Text context after the word
            
        Returns:
            List of (candidate, score) tuples, sorted by score
        """
        # Create masked sentence for each candidate
        masked_text = f"{context_before} {self.mask_token} {context_after}"
        
        # Tokenize input
        inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
        
        # Find position of mask token
        mask_positions = (inputs.input_ids == self.mask_token_id).nonzero(as_tuple=True)[1]
        
        if len(mask_positions) == 0:
            # No mask token found, return candidates with original scores
            return [(candidate, 0.0) for candidate in candidates]
        
        mask_position = mask_positions[0].item()
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits[0, mask_position]
            
        # Calculate scores for each candidate
        scores = []
        for candidate in candidates:
            # Tokenize candidate (handle multi-token words)
            candidate_tokens = self.tokenizer.tokenize(candidate)
            
            # If candidate is split into multiple tokens, average the scores
            if len(candidate_tokens) == 1:
                # Single token
                token_id = self.tokenizer.convert_tokens_to_ids(candidate_tokens[0])
                score = predictions[token_id].item()
                scores.append((candidate, score))
            else:
                # Multi-token word
                # This is a simplification - a better approach would handle multi-token words more carefully
                token_ids = self.tokenizer.convert_tokens_to_ids(candidate_tokens)
                score = np.mean([predictions[token_id].item() for token_id in token_ids])
                scores.append((candidate, score))
        
        # Normalize scores using softmax
        scores_array = np.array([score for _, score in scores])
        exp_scores = np.exp(scores_array - np.max(scores_array))
        softmax_scores = exp_scores / exp_scores.sum()
        
        # Create final ranked list
        ranked_candidates = [(candidates[i], softmax_scores[i]) for i in range(len(candidates))]
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_candidates
    
    def rerank_with_bilingual_context(self,
                                     candidates: List[str],
                                     thai_context: str,
                                     english_partial_translation: str) -> List[Tuple[str, float]]:
        """
        Rerank candidates using both Thai and English context
        
        Args:
            candidates: List of English word candidates
            thai_context: Original Thai context
            english_partial_translation: Partial English translation
            
        Returns:
            List of (candidate, score) tuples, sorted by score
        """
        # This is a more advanced method that would use a bilingual model
        # For simplicity, we'll just use the English context
        
        # Extract context before and after from partial translation
        # This is a placeholder - actual implementation would be more sophisticated
        parts = english_partial_translation.split('[MASK]')
        
        if len(parts) != 2:
            # Fallback to simple scoring
            return [(candidate, 1.0/idx) for idx, candidate in enumerate(candidates, 1)]
            
        context_before, context_after = parts[0], parts[1]
        
        # Use monolingual reranking
        return self.rerank_candidates(candidates, context_before, context_after)