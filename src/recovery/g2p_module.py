import re
import json
import os
from typing import List, Dict, Tuple, Optional, Set, Union
import numpy as np

class ThaiG2P:
    """
    Thai Grapheme-to-Phoneme converter
    Converts Thai characters to phonetic representations (IPA or custom)
    """
    def __init__(self, 
                 phoneme_map_path: Optional[str] = None,
                 use_ipa: bool = True):
        """
        Initialize the Thai G2P converter
        
        Args:
            phoneme_map_path: Path to custom phoneme mapping file (JSON)
            use_ipa: Whether to use IPA (International Phonetic Alphabet)
        """
        self.use_ipa = use_ipa
        
        # Load phoneme mappings
        if phoneme_map_path and os.path.exists(phoneme_map_path):
            with open(phoneme_map_path, 'r', encoding='utf-8') as f:
                self.phoneme_map = json.load(f)
        else:
            # Default phoneme mappings (simplified)
            self.phoneme_map = self._get_default_phoneme_map()
            
        # Compile regular expressions for syllable patterns
        self._compile_syllable_patterns()
        
    def _get_default_phoneme_map(self) -> Dict[str, str]:
        """
        Get default phoneme mappings for Thai characters
        
        Returns:
            Dictionary mapping Thai characters to phonetic representations
        """
        # This is a simplified mapping - a complete implementation would be more extensive
        consonants = {
            # Initial consonants
            'ก': 'k', 'ข': 'kh', 'ฃ': 'kh', 'ค': 'kh', 'ฅ': 'kh', 'ฆ': 'kh',
            'ง': 'ŋ', 'จ': 'tɕ', 'ฉ': 'tɕh', 'ช': 'tɕh', 'ซ': 's', 'ฌ': 'tɕh',
            'ญ': 'j', 'ฎ': 'd', 'ฏ': 't', 'ฐ': 'th', 'ฑ': 'th', 'ฒ': 'th',
            'ณ': 'n', 'ด': 'd', 'ต': 't', 'ถ': 'th', 'ท': 'th', 'ธ': 'th',
            'น': 'n', 'บ': 'b', 'ป': 'p', 'ผ': 'ph', 'ฝ': 'f', 'พ': 'ph',
            'ฟ': 'f', 'ภ': 'ph', 'ม': 'm', 'ย': 'j', 'ร': 'r', 'ล': 'l',
            'ว': 'w', 'ศ': 's', 'ษ': 's', 'ส': 's', 'ห': 'h', 'ฬ': 'l',
            'อ': 'ʔ', 'ฮ': 'h',
            
            # Final consonants (different pronunciation)
            'ก_final': 'k̚', 'ข_final': 'k̚', 'ค_final': 'k̚', 'ง_final': 'ŋ',
            'จ_final': 't̚', 'ด_final': 't̚', 'ต_final': 't̚', 'ท_final': 't̚',
            'น_final': 'n', 'บ_final': 'p̚', 'ป_final': 'p̚', 'ม_final': 'm',
            'ย_final': 'j', 'ว_final': 'w'
        }
        
        vowels = {
            # Short vowels
            'ะ': 'a', 'ิ': 'i', 'ึ': 'ɯ', 'ุ': 'u', 'เะ': 'e', 'แะ': 'ɛ',
            'โะ': 'o', 'เาะ': 'ɔ', 'ัะ': 'a',
            
            # Long vowels
            'า': 'aː', 'ี': 'iː', 'ื': 'ɯː', 'ู': 'uː', 'เ': 'eː', 'แ': 'ɛː',
            'โ': 'oː', 'อ': 'ɔː', 'ั': 'a',
            
            # Diphthongs
            'เียะ': 'ia', 'เือะ': 'ɯa', 'ัวะ': 'ua',
            'เีย': 'iaː', 'เือ': 'ɯaː', 'ัว': 'uaː',
            'ไ': 'aj', 'ใ': 'aj', 'ไย': 'aj', 'าย': 'aːj',
            'อย': 'ɔːj', 'โย': 'oːj', 'ุย': 'uj', 'เย': 'əːj',
            'วย': 'uaj', 'าว': 'aːw', 'เา': 'aw', 'แว': 'ɛːw',
            'เว': 'eːw', 'เียว': 'iaw', 'ิว': 'iw'
        }
        
        tones = {
            '่': '˩', '้': '˥˩', '๊': '˦', '๋': '˥', '': '˧'  # Mid tone (unmarked)
        }
        
        # Combine all mappings
        phoneme_map = {}
        phoneme_map.update(consonants)
        phoneme_map.update(vowels)
        phoneme_map.update(tones)
        
        return phoneme_map
    
    def _compile_syllable_patterns(self):
        """Compile regular expressions for Thai syllable patterns"""
        # This is a simplified approach - a complete implementation would be more complex
        # Thai syllable structure: C(C)V(C)T where C=consonant, V=vowel, T=tone
        
        # Define character classes
        consonants = r'[กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ]'
        vowels = r'[ะิึุเแโาีืูัำๅ]'
        tones = r'[่้๊๋]?'
        
        # Define syllable patterns (simplified)
        self.syllable_patterns = [
            # CV pattern
            re.compile(f'{consonants}{vowels}{tones}'),
            # CVC pattern
            re.compile(f'{consonants}{vowels}{consonants}{tones}'),
            # CCVC pattern
            re.compile(f'{consonants}{consonants}{vowels}{consonants}{tones}')
        ]
    
    def convert_to_phonemes(self, text: str) -> str:
        """
        Convert Thai text to phonetic representation
        
        Args:
            text: Thai text to convert
            
        Returns:
            Phonetic representation of the text
        """
        # Segment text into syllables
        syllables = self._segment_syllables(text)
        
        # Convert each syllable to phonemes
        phonemes = []
        for syllable in syllables:
            syllable_phonemes = self._convert_syllable(syllable)
            phonemes.append(syllable_phonemes)
        
        # Join phonemes with appropriate separators
        if self.use_ipa:
            # Use IPA conventions
            result = '.'.join(phonemes)
        else:
            # Use custom separator
            result = ' '.join(phonemes)
            
        return result
    
    def _segment_syllables(self, text: str) -> List[str]:
        """
        Segment Thai text into syllables
        
        Args:
            text: Thai text to segment
            
        Returns:
            List of syllables
        """
        # This is a simplified approach - a complete implementation would use
        # more sophisticated syllable segmentation algorithms
        
        # For demonstration, we'll use a greedy approach
        syllables = []
        remaining_text = text
        
        while remaining_text:
            match_found = False
            
            # Try to match syllable patterns
            for pattern in self.syllable_patterns:
                match = pattern.match(remaining_text)
                if match:
                    syllable = match.group(0)
                    syllables.append(syllable)
                    remaining_text = remaining_text[len(syllable):]
                    match_found = True
                    break
            
            # If no pattern matches, take one character as a syllable
            if not match_found:
                syllables.append(remaining_text[0])
                remaining_text = remaining_text[1:]
        
        return syllables
    
    def _convert_syllable(self, syllable: str) -> str:
        """
        Convert a Thai syllable to phonetic representation
        
        Args:
            syllable: Thai syllable to convert
            
        Returns:
            Phonetic representation of the syllable
        """
        # This is a simplified approach - a complete implementation would handle
        # all the complexities of Thai phonology
        
        # Extract components (initial consonant, vowel, final consonant, tone)
        # This is a placeholder - actual implementation would be more complex
        initial_consonant = syllable[0] if syllable else ''
        vowel = ''
        final_consonant = ''
        tone = ''
        
        # Look up phonemes in the mapping
        initial_phoneme = self.phoneme_map.get(initial_consonant, initial_consonant)
        vowel_phoneme = self.phoneme_map.get(vowel, vowel)
        final_phoneme = self.phoneme_map.get(f"{final_consonant}_final", final_consonant)
        tone_phoneme = self.phoneme_map.get(tone, '')
        
        # Combine phonemes according to phonological rules
        syllable_phoneme = f"{initial_phoneme}{vowel_phoneme}{final_phoneme}{tone_phoneme}"
        
        return syllable_phoneme