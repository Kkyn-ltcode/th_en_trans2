import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any, Union

from ..detection.base_detector import TransliterationCandidate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TransliterationResult:
    """
    Class representing a transliteration recovery result
    """
    def __init__(self, 
                 original_token: str,
                 english_candidates: List[str],
                 best_candidate: str,
                 confidence: float,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a transliteration result
        
        Args:
            original_token: The original Thai transliterated token
            english_candidates: List of potential English candidates
            best_candidate: The best English candidate selected
            confidence: Confidence score for the best candidate (0.0 to 1.0)
            metadata: Additional metadata about the recovery process
        """
        self.original_token = original_token
        self.english_candidates = english_candidates
        self.best_candidate = best_candidate
        self.confidence = confidence
        self.metadata = metadata or {}
        
    def __repr__(self) -> str:
        return f"TransliterationResult('{self.original_token}' → '{self.best_candidate}', confidence={self.confidence:.2f})"

class BaseRecoveryModule(ABC):
    """
    Abstract base class for transliteration recovery modules
    """
    def __init__(self, name: str = "base_recovery"):
        """
        Initialize the recovery module
        
        Args:
            name: Name of the module for logging and identification
        """
        self.name = name
        self.logger = logging.getLogger(f"recovery.{name}")
        
    @abstractmethod
    def recover(self, candidate: TransliterationCandidate) -> TransliterationResult:
        """
        Recover the original English form of a transliterated Thai token
        
        Args:
            candidate: The transliteration candidate to recover
            
        Returns:
            A TransliterationResult object with the recovered English form
        """
        pass
    
    @abstractmethod
    def recover_batch(self, candidates: List[TransliterationCandidate]) -> List[TransliterationResult]:
        """
        Recover the original English forms for a batch of transliteration candidates
        
        Args:
            candidates: List of transliteration candidates to recover
            
        Returns:
            List of TransliterationResult objects
        """
        pass