from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TransliterationCandidate:
    """
    Class representing a transliteration candidate
    """
    def __init__(self, 
                 token: str, 
                 start_pos: int, 
                 end_pos: int, 
                 confidence: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a transliteration candidate
        
        Args:
            token: The token text
            start_pos: Start position in the original text
            end_pos: End position in the original text
            confidence: Confidence score (0.0 to 1.0)
            metadata: Additional metadata about the candidate
        """
        self.token = token
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.confidence = confidence
        self.metadata = metadata or {}
        
    def __repr__(self) -> str:
        return f"TransliterationCandidate(token='{self.token}', confidence={self.confidence:.2f})"

class BaseCandidateDetector(ABC):
    """
    Abstract base class for transliteration candidate detectors
    """
    def __init__(self, name: str = "base_detector"):
        """
        Initialize the detector
        
        Args:
            name: Name of the detector for logging and identification
        """
        self.name = name
        self.logger = logging.getLogger(f"detector.{name}")
        
    @abstractmethod
    def detect_candidates(self, text: str) -> List[TransliterationCandidate]:
        """
        Detect transliteration candidates in the given text
        
        Args:
            text: The Thai text to analyze
            
        Returns:
            A list of TransliterationCandidate objects
        """
        pass
    
    def filter_candidates(self, 
                         candidates: List[TransliterationCandidate], 
                         min_confidence: float = 0.5) -> List[TransliterationCandidate]:
        """
        Filter candidates based on confidence threshold
        
        Args:
            candidates: List of candidates to filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of candidates
        """
        return [c for c in candidates if c.confidence >= min_confidence]