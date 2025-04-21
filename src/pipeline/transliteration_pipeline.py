import os
import logging
from typing import List, Dict, Tuple, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import detection modules
from ..utils.tokenizer import ThaiTokenizer
from ..detection.base_detector import TransliterationCandidate
from ..detection.rule_based_detector import RuleBasedDetector
from ..detection.ngram_detector import NgramDetector
from ..detection.ensemble_detector import EnsembleDetector

# Import recovery modules
from ..recovery.base_recovery import TransliterationResult
from ..recovery.dictionary_recovery import DictionaryRecovery
from ..recovery.neural_recovery import NeuralRecovery
from ..recovery.ensemble_recovery import EnsembleRecovery

class TransliterationPipeline:
    """
    Complete pipeline for Thai-English transliteration detection and recovery
    """
    def __init__(self, 
                 detector_type: str = "ensemble",
                 recovery_type: str = "ensemble",
                 dictionary_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 use_context: bool = True):
        """
        Initialize the transliteration pipeline
        
        Args:
            detector_type: Type of detector to use ('rule', 'ngram', 'ensemble')
            recovery_type: Type of recovery to use ('dictionary', 'neural', 'ensemble')
            dictionary_path: Path to Thai-English transliteration dictionary
            model_path: Path to neural model for recovery
            use_context: Whether to use context for recovery
        """
        self.logger = logging.getLogger("transliteration_pipeline")
        self.use_context = use_context
        
        # Initialize Thai tokenizer
        self.tokenizer = ThaiTokenizer()
        
        # Initialize detector
        self._initialize_detector(detector_type)
        
        # Initialize recovery module
        self._initialize_recovery(recovery_type, dictionary_path, model_path)
        
        self.logger.info(f"Initialized transliteration pipeline with {detector_type} detector and {recovery_type} recovery")
    
    def _initialize_detector(self, detector_type: str):
        """Initialize the appropriate detector"""
        if detector_type == "rule":
            self.detector = RuleBasedDetector(tokenizer=self.tokenizer)
        elif detector_type == "ngram":
            self.detector = NgramDetector(tokenizer=self.tokenizer)
        elif detector_type == "ensemble":
            # Create individual detectors for ensemble
            rule_detector = RuleBasedDetector(tokenizer=self.tokenizer)
            ngram_detector = NgramDetector(tokenizer=self.tokenizer)
            
            # Create ensemble detector
            self.detector = EnsembleDetector(
                detectors=[rule_detector, ngram_detector],
                weights=[0.6, 0.4]  # Giving more weight to rule-based detector
            )
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    def _initialize_recovery(self, recovery_type: str, dictionary_path: Optional[str], model_path: Optional[str]):
        """Initialize the appropriate recovery module"""
        if recovery_type == "dictionary":
            self.recovery = DictionaryRecovery(dictionary_path=dictionary_path)
        elif recovery_type == "neural":
            model_path = model_path or "google/mt5-small"  # Default model if none provided
            self.recovery = NeuralRecovery(model_name_or_path=model_path)
        elif recovery_type == "ensemble":
            # Create individual recovery modules for ensemble
            dict_recovery = DictionaryRecovery(dictionary_path=dictionary_path)
            
            # For neural recovery, use a simpler model if no path provided
            model_path = model_path or "google/mt5-small"
            neural_recovery = NeuralRecovery(model_name_or_path=model_path)
            
            # Create ensemble recovery
            self.recovery = EnsembleRecovery(
                recovery_modules=[dict_recovery, neural_recovery],
                weights=[0.6, 0.4],  # Giving more weight to dictionary recovery
                use_context_reranking=self.use_context
            )
        else:
            raise ValueError(f"Unknown recovery type: {recovery_type}")
    
    def process_text(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process Thai text to detect and recover transliterations
        
        Args:
            text: Thai text to process
            
        Returns:
            Tuple of (processed_text, transliteration_info)
            - processed_text: Text with transliterations replaced by English
            - transliteration_info: List of dictionaries with info about each transliteration
        """
        # Step 1: Detect transliteration candidates
        candidates = self.detector.detect_candidates(text)
        
        if not candidates:
            return text, []
        
        # Step 2: Recover original English forms
        results = self.recovery.recover_batch(candidates)
        
        # Step 3: Replace transliterations in the original text
        # Sort candidates by position (from right to left to avoid offset issues)
        sorted_candidates_results = sorted(
            zip(candidates, results),
            key=lambda x: x[0].start_pos,
            reverse=True
        )
        
        # Make a copy of the original text
        processed_text = text
        
        # Information about each transliteration
        transliteration_info = []
        
        # Replace each transliteration
        for candidate, result in sorted_candidates_results:
            # Skip if no English candidate was found or confidence is too low
            if not result.best_candidate or result.confidence < 0.5:
                continue
            
            # Replace the Thai transliteration with the English word
            start, end = candidate.start_pos, candidate.end_pos
            
            # Store information about this transliteration
            info = {
                "thai": candidate.token,
                "english": result.best_candidate,
                "position": (start, end),
                "confidence": result.confidence,
                "alternatives": result.english_candidates
            }
            transliteration_info.append(info)
            
            # Replace in the text
            processed_text = processed_text[:start] + result.best_candidate + processed_text[end:]
        
        return processed_text, transliteration_info
    
    def process_text_with_markup(self, text: str, markup_format: str = "html") -> str:
        """
        Process Thai text and return with markup highlighting transliterations
        
        Args:
            text: Thai text to process
            markup_format: Format for markup ('html', 'markdown', 'console')
            
        Returns:
            Text with transliterations marked up
        """
        # Process the text
        _, transliteration_info = self.process_text(text)
        
        if not transliteration_info:
            return text
        
        # Sort by position (from right to left to avoid offset issues)
        sorted_info = sorted(transliteration_info, key=lambda x: x["position"][0], reverse=True)
        
        # Make a copy of the original text
        marked_text = text
        
        # Apply markup based on the requested format
        for info in sorted_info:
            start, end = info["position"]
            thai = info["thai"]
            english = info["english"]
            
            if markup_format == "html":
                replacement = f'<span class="transliteration" title="{english}">{thai}</span>'
            elif markup_format == "markdown":
                replacement = f'{thai} [{english}]'
            elif markup_format == "console":
                replacement = f'{thai} ({english})'
            else:
                replacement = f'{thai} -> {english}'
            
            marked_text = marked_text[:start] + replacement + marked_text[end:]
        
        return marked_text
    
    def process_text_with_replacement(self, text: str, replacement_format: str = "{english}") -> str:
        """
        Process Thai text and replace transliterations according to format
        
        Args:
            text: Thai text to process
            replacement_format: Format string for replacement
                                Use {thai} for original Thai, {english} for English
            
        Returns:
            Text with transliterations replaced according to format
        """
        # Process the text
        _, transliteration_info = self.process_text(text)
        
        if not transliteration_info:
            return text
        
        # Sort by position (from right to left to avoid offset issues)
        sorted_info = sorted(transliteration_info, key=lambda x: x["position"][0], reverse=True)
        
        # Make a copy of the original text
        replaced_text = text
        
        # Replace each transliteration
        for info in sorted_info:
            start, end = info["position"]
            thai = info["thai"]
            english = info["english"]
            
            # Format the replacement
            replacement = replacement_format.format(thai=thai, english=english)
            
            # Replace in the text
            replaced_text = replaced_text[:start] + replacement + replaced_text[end:]
        
        return replaced_text
    
    def train_detector(self, thai_texts: List[str], transliterated_texts: List[str]):
        """
        Train the detector component (if applicable)
        
        Args:
            thai_texts: List of Thai texts without transliterations
            transliterated_texts: List of Thai texts with transliterations
        """
        # Check if detector supports training
        if hasattr(self.detector, 'train'):
            self.detector.train(thai_texts, transliterated_texts)
            self.logger.info("Trained detector successfully")
        else:
            self.logger.warning("Current detector does not support training")
    
    def fine_tune_recovery(self, 
                          thai_words: List[str], 
                          english_words: List[str],
                          output_dir: str = "models/fine_tuned_transliteration"):
        """
        Fine-tune the recovery component (if applicable)
        
        Args:
            thai_words: List of Thai transliterated words
            english_words: List of corresponding English words
            output_dir: Directory to save fine-tuned model
        """
        # Check if recovery supports fine-tuning
        if hasattr(self.recovery, 'fine_tune'):
            self.recovery.fine_tune(thai_words, english_words, output_dir=output_dir)
            self.logger.info(f"Fine-tuned recovery model saved to {output_dir}")
        else:
            self.logger.warning("Current recovery module does not support fine-tuning")