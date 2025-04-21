import re
import pythainlp
from pythainlp.tokenize import word_tokenize
from typing import List, Tuple, Dict, Any, Optional

class ThaiTokenizer:
    """
    Tokenizer for Thai text that handles various tokenization strategies
    """
    def __init__(self, engine: str = "newmm"):
        """
        Initialize the tokenizer
        
        Args:
            engine: The tokenization engine to use (default: newmm)
                   Options include 'newmm', 'longest', 'icu', etc.
        """
        self.engine = engine
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Thai text into words
        
        Args:
            text: The Thai text to tokenize
            
        Returns:
            A list of tokens
        """
        # Pre-process text to handle special cases
        text = self._preprocess_text(text)
        
        # Tokenize using pythainlp
        tokens = word_tokenize(text, engine=self.engine)
        
        return tokens
    
    def tokenize_with_positions(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Tokenize Thai text and return tokens with their positions
        
        Args:
            text: The Thai text to tokenize
            
        Returns:
            A list of tuples (token, start_position, end_position)
        """
        tokens = self.tokenize(text)
        
        # Track positions
        positions = []
        current_pos = 0
        
        for token in tokens:
            token_start = text.find(token, current_pos)
            if token_start == -1:
                # Handle case where token can't be found directly
                # This can happen with some tokenizers that transform tokens
                token_start = current_pos
                
            token_end = token_start + len(token)
            positions.append((token, token_start, token_end))
            current_pos = token_end
            
        return positions
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before tokenization
        
        Args:
            text: The text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Handle Thai-specific typography if needed
        # Add more preprocessing as needed
        
        return text