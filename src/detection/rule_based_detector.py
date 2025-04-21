import re
from typing import List, Set, Dict, Tuple, Optional
from .base_detector import BaseCandidateDetector, TransliterationCandidate
from ..utils.tokenizer import ThaiTokenizer

class RuleBasedDetector(BaseCandidateDetector):
    """
    Rule-based detector for transliteration candidates using heuristics
    """
    def __init__(self, 
                 tokenizer: Optional[ThaiTokenizer] = None,
                 name: str = "rule_based"):
        """
        Initialize the rule-based detector
        
        Args:
            tokenizer: ThaiTokenizer instance (creates one if None)
            name: Name of the detector
        """
        super().__init__(name=name)
        self.tokenizer = tokenizer or ThaiTokenizer()
        
        # Initialize rules
        self._initialize_rules()
        
    def _initialize_rules(self):
        """Initialize detection rules and patterns"""
        # Common transliteration patterns
        self.trans_patterns = [
            # Common ending patterns in transliterations
            r'เซอร์$', r'ชั่น$', r'ด์$', r'ติ้ง$', r'ไซส์$',
            # Common consonant clusters not typical in Thai
            r'[ฟซคท]ล', r'[บพ]ร', r'[ดท]ร',
            # Vowel patterns common in transliterations
            r'โอ[^าีิ]', r'เอ[^าีิ]',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern) for pattern in self.trans_patterns]
        
        # Common transliteration syllables
        self.trans_syllables = {
            'คอม', 'ไมค์', 'ไลฟ์', 'ไลน์', 'เกม', 'เน็ต', 'เซิร์ฟ', 
            'เวอร์', 'เซอร์', 'เทค', 'โปร', 'กราม', 'ดอท', 'ช็อป',
            'ชิพ', 'ทวิต', 'เตอร์', 'เฟส', 'บุ๊ค', 'อิน', 'สตา', 'แกรม'
        }
        
        # Rare Thai character combinations that often indicate transliterations
        self.rare_combinations = {
            'ฟต', 'ฟร', 'ฟล', 'พร', 'ดร', 'ทร', 'ฟอ', 'ฟา', 'ชั่น'
        }
        
    def detect_candidates(self, text: str) -> List[TransliterationCandidate]:
        """
        Detect transliteration candidates using rule-based heuristics
        
        Args:
            text: Thai text to analyze
            
        Returns:
            List of transliteration candidates
        """
        candidates = []
        
        # Tokenize with positions
        token_positions = self.tokenizer.tokenize_with_positions(text)
        
        for token, start_pos, end_pos in token_positions:
            confidence = self._calculate_confidence(token)
            
            if confidence > 0:
                metadata = {
                    "detection_method": "rule_based",
                    "rules_matched": self._get_matched_rules(token)
                }
                
                candidate = TransliterationCandidate(
                    token=token,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    confidence=confidence,
                    metadata=metadata
                )
                
                candidates.append(candidate)
                
        return candidates
    
    def _calculate_confidence(self, token: str) -> float:
        """
        Calculate confidence score for a token based on rules
        
        Args:
            token: The token to evaluate
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Skip very short tokens
        if len(token) < 2:
            return 0.0
            
        # Initialize score
        score = 0.0
        
        # Check for exact matches in transliteration syllables
        if token in self.trans_syllables:
            score += 0.7
            
        # Check for rare combinations
        for combo in self.rare_combinations:
            if combo in token:
                score += 0.4
                break
                
        # Check regex patterns
        for pattern in self.compiled_patterns:
            if pattern.search(token):
                score += 0.5
                break
                
        # Penalize very common Thai words (would need a frequency list)
        # This is a placeholder for demonstration
        common_thai_words = {'และ', 'ที่', 'ของ', 'ใน', 'การ', 'ไป', 'มา', 'ให้'}
        if token in common_thai_words:
            score -= 0.8
            
        # Normalize score to 0.0-1.0 range
        score = max(0.0, min(1.0, score))
        
        return score
    
    def _get_matched_rules(self, token: str) -> List[str]:
        """
        Get list of rules that matched for this token
        
        Args:
            token: The token to check
            
        Returns:
            List of rule names that matched
        """
        matched = []
        
        if token in self.trans_syllables:
            matched.append("common_transliteration_syllable")
            
        for combo in self.rare_combinations:
            if combo in token:
                matched.append(f"rare_combination:{combo}")
                break
                
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(token):
                matched.append(f"pattern:{self.trans_patterns[i]}")
                
        return matched