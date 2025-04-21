import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ..detection.base_detector import TransliterationCandidate
from .base_recovery import BaseRecoveryModule, TransliterationResult

class NeuralRecovery(BaseRecoveryModule):
    """
    Neural model-based transliteration recovery module
    """
    def __init__(self, 
                 model_name_or_path: str = "google/mt5-small",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 name: str = "neural_recovery"):
        """
        Initialize the neural recovery module
        
        Args:
            model_name_or_path: Pretrained model name or path
            device: Device to run inference on ('cuda', 'cpu', etc.)
            cache_dir: Directory to cache models
            name: Name of the module
        """
        super().__init__(name=name)
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Loaded model from {model_name_or_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def recover(self, candidate: TransliterationCandidate) -> TransliterationResult:
        """
        Recover the original English form using neural model
        
        Args:
            candidate: The transliteration candidate to recover
            
        Returns:
            A TransliterationResult object with the recovered English form
        """
        thai_token = candidate.token
        
        # Prepare input for the model
        # Format: "transliterate: [THAI_WORD]"
        input_text = f"transliterate: {thai_token}"
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Generate predictions
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=50,
                num_beams=5,
                num_return_sequences=5,
                early_stopping=True
            )
        
        # Decode predictions
        english_candidates = []
        for output in outputs:
            english = self.tokenizer.decode(output, skip_special_tokens=True)
            english_candidates.append(english)
        
        # If no candidates were generated, return empty result
        if not english_candidates:
            return TransliterationResult(
                original_token=thai_token,
                english_candidates=[],
                best_candidate="",
                confidence=0.0,
                metadata={"source": "neural_model_failed"}
            )
        
        # Best candidate is the first one (highest probability)
        best_candidate = english_candidates[0]
        
        # Calculate confidence based on model output
        # This is a simplified approach - a real implementation would use the model's confidence scores
        confidence = 0.9  # Default high confidence for neural model
        
        return TransliterationResult(
            original_token=thai_token,
            english_candidates=english_candidates,
            best_candidate=best_candidate,
            confidence=confidence,
            metadata={
                "source": "neural_model",
                "model_name": self.model.config._name_or_path
            }
        )
    
    def recover_batch(self, candidates: List[TransliterationCandidate]) -> List[TransliterationResult]:
        """
        Recover the original English forms for a batch of transliteration candidates
        
        Args:
            candidates: List of transliteration candidates to recover
            
        Returns:
            List of TransliterationResult objects
        """
        # For small batches, process individually
        if len(candidates) <= 4:
            return [self.recover(candidate) for candidate in candidates]
        
        # For larger batches, process in batch mode
        thai_tokens = [candidate.token for candidate in candidates]
        
        # Prepare batch input
        input_texts = [f"transliterate: {token}" for token in thai_tokens]
        
        # Tokenize inputs
        batch_inputs = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        # Generate predictions
        with torch.no_grad():
            batch_outputs = self.model.generate(
                batch_inputs.input_ids,
                max_length=50,
                num_beams=3,  # Reduced beam size for batch processing
                num_return_sequences=1,  # Only return best sequence for batch
                early_stopping=True
            )
        
        # Decode predictions
        batch_results = []
        for i, output in enumerate(batch_outputs):
            english = self.tokenizer.decode(output, skip_special_tokens=True)
            
            result = TransliterationResult(
                original_token=thai_tokens[i],
                english_candidates=[english],
                best_candidate=english,
                confidence=0.9,  # Default confidence
                metadata={
                    "source": "neural_model_batch",
                    "model_name": self.model.config._name_or_path
                }
            )
            batch_results.append(result)
        
        return batch_results
    
    def fine_tune(self, 
                 thai_words: List[str], 
                 english_words: List[str],
                 epochs: int = 3,
                 batch_size: int = 16,
                 learning_rate: float = 5e-5,
                 output_dir: str = "models/fine_tuned_transliteration"):
        """
        Fine-tune the model on transliteration data
        
        Args:
            thai_words: List of Thai transliterated words
            english_words: List of corresponding English words
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            output_dir: Directory to save fine-tuned model
        """
        from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
        import datasets
        
        # Prepare dataset
        train_data = {
            "thai": thai_words,
            "english": english_words
        }
        
        # Create dataset
        dataset = datasets.Dataset.from_dict(train_data)
        
        # Tokenize dataset
        def preprocess_function(examples):
            inputs = [f"transliterate: {thai}" for thai in examples["thai"]]
            targets = examples["english"]
            
            model_inputs = self.tokenizer(inputs, max_length=128, truncation=True)
            labels = self.tokenizer(targets, max_length=128, truncation=True)
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            save_total_limit=2,
            predict_with_generate=True,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_strategy="epoch"
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train model
        self.logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.logger.info(f"Model fine-tuned and saved to {output_dir}")