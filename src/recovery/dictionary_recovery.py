import os
import json
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict

from ..detection.base_detector import TransliterationCandidate
from .base_recovery import BaseRecoveryModule, TransliterationResult

class DictionaryRecovery(BaseRecoveryModule):
    """
    Dictionary-based transliteration recovery module
    """
    def __init__(self, 
                 dictionary_path: Optional[str] = None,
                 name: str = "dictionary_recovery"):
        """
        Initialize the dictionary-based recovery module
        
        Args:
            dictionary_path: Path to Thai-English transliteration dictionary
            name: Name of the module
        """
        super().__init__(name=name)
        
        # Initialize dictionaries
        self.thai_to_english = {}
        self.english_to_thai = defaultdict(list)
        
        # Load dictionary if provided
        if dictionary_path and os.path.exists(dictionary_path):
            self._load_dictionary(dictionary_path)
        else:
            self.logger.warning("No dictionary provided. Using empty dictionary.")
    
    def _load_dictionary(self, path: str):
        """
        Load Thai-English transliteration dictionary
        
        Args:
            path: Path to the dictionary file
        """
        # Determine file format based on extension
        _, ext = os.path.splitext(path)
        
        if ext.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                self.thai_to_english = json.load(f)
        else:
            # Assume text format with thai<tab>english
            self.thai_to_english = {}
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        thai, english = parts[0], parts[1]
                        self.thai_to_english[thai] = english
        
        # Build reverse mapping
        for thai, english in self.thai_to_english.items():
            self.english_to_thai[english].append(thai)
            
        self.logger.info(f"Loaded {len(self.thai_to_english)} entries from dictionary")
    
    def recover(self, candidate: TransliterationCandidate) -> TransliterationResult:
        """
        Recover the original English form using dictionary lookup
        
        Args:
            candidate: The transliteration candidate to recover
            
        Returns:
            A TransliterationResult object with the recovered English form
        """
        thai_token = candidate.token
        
        # Direct dictionary lookup
        if thai_token in self.thai_to_english:
            english_word = self.thai_to_english[thai_token]
            return TransliterationResult(
                original_token=thai_token,
                english_candidates=[english_word],
                best_candidate=english_word,
                confidence=1.0,
                metadata={"source": "direct_dictionary_lookup"}
            )
        
        # Try case variations
        english_candidates = []
        
        # Try lowercase
        lowercase_token = thai_token.lower()
        if lowercase_token != thai_token and lowercase_token in self.thai_to_english:
            english_candidates.append(self.thai_to_english[lowercase_token])
        
        # If we found candidates, return the best one
        if english_candidates:
            best_candidate = english_candidates[0]
            return TransliterationResult(
                original_token=thai_token,
                english_candidates=english_candidates,
                best_candidate=best_candidate,
                confidence=0.9,  # Slightly lower confidence for case variations
                metadata={"source": "case_variation_lookup"}
            )
        
        # Try fuzzy matching for similar Thai words
        similar_candidates = self._find_similar_thai_words(thai_token)
        if similar_candidates:
            # Get English translations for similar Thai words
            english_candidates = []
            for similar_thai, similarity in similar_candidates:
                if similar_thai in self.thai_to_english:
                    english = self.thai_to_english[similar_thai]
                    english_candidates.append((english, similarity))
            
            if english_candidates:
                # Sort by similarity
                english_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Extract just the English words
                english_words = [word for word, _ in english_candidates]
                best_candidate = english_words[0]
                
                # Calculate confidence based on similarity
                confidence = english_candidates[0][1]
                
                return TransliterationResult(
                    original_token=thai_token,
                    english_candidates=english_words,
                    best_candidate=best_candidate,
                    confidence=confidence,
                    metadata={"source": "fuzzy_dictionary_lookup"}
                )
        
        # No match found
        return TransliterationResult(
            original_token=thai_token,
            english_candidates=[],
            best_candidate="",
            confidence=0.0,
            metadata={"source": "dictionary_lookup_failed"}
        )
    
    def _find_similar_thai_words(self, thai_word: str, max_candidates: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar Thai words in the dictionary
        
        Args:
            thai_word: Thai word to find similar words for
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of (similar_word, similarity_score) tuples
        """
        candidates = []
        
        # Simple character-based similarity
        for dict_word in self.thai_to_english.keys():
            # Skip exact match (already handled)
            if dict_word == thai_word:
                continue
                
            # Calculate similarity
            similarity = self._calculate_string_similarity(thai_word, dict_word)
            
            # Add if similarity is above threshold
            if similarity > 0.7:
                candidates.append((dict_word, similarity))
        
        # Sort by similarity and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Levenshtein distance-based similarity
        # This is a simple implementation - a production system would use a more efficient algorithm
        
        # Initialize matrix
        m, n = len(str1), len(str2)
        if m == 0 or n == 0:
            return 0.0
            
        matrix = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Fill matrix
        for i in range(m + 1):
            matrix[i][0] = i
        for j in range(n + 1):
            matrix[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if str1[i-1] == str2[j-1] else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # Deletion
                    matrix[i][j-1] + 1,      # Insertion
                    matrix[i-1][j-1] + cost  # Substitution
                )
        
        # Calculate similarity from distance
        distance = matrix[m][n]
        max_len = max(m, n)
        similarity = 1.0 - (distance / max_len)
        
        return similarity
    
    def recover_batch(self, candidates: List[TransliterationCandidate]) -> List[TransliterationResult]:
        """
        Recover the original English forms for a batch of transliteration candidates
        
        Args:
            candidates: List of transliteration candidates to recover
            
        Returns:
            List of TransliterationResult objects
        """
        results = []
        for candidate in candidates:
            result = self.recover(candidate)
            results.append(result)
        return results