from typing import List, Dict, Tuple, Optional, Union
from .base_detector import BaseCandidateDetector, TransliterationCandidate
from ..utils.tokenizer import ThaiTokenizer

class EnsembleDetector(BaseCandidateDetector):
    """
    Ensemble detector that combines multiple detection approaches
    """
    def __init__(self, 
                 detectors: List[BaseCandidateDetector],
                 weights: Optional[List[float]] = None,
                 name: str = "ensemble_detector"):
        """
        Initialize the ensemble detector
        
        Args:
            detectors: List of detector instances to ensemble
            weights: List of weights for each detector (must match detectors length)
            name: Name of the detector
        """
        super().__init__(name=name)
        self.detectors = detectors
        
        # Validate and normalize weights
        if weights is None:
            # Equal weights if not provided
            self.weights = [1.0 / len(detectors)] * len(detectors)
        else:
            if len(weights) != len(detectors):
                raise ValueError("Number of weights must match number of detectors")
            
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
        self.logger.info(f"Initialized ensemble with {len(detectors)} detectors")
        
    def detect_candidates(self, text: str) -> List[TransliterationCandidate]:
        """
        Detect transliteration candidates using all detectors in the ensemble
        
        Args:
            text: Thai text to analyze
            
        Returns:
            List of transliteration candidates with combined confidence scores
        """
        all_candidates = []
        
        # Collect candidates from all detectors
        for i, detector in enumerate(self.detectors):
            detector_candidates = detector.detect_candidates(text)
            
            # Add detector info to metadata
            for candidate in detector_candidates:
                if "source_detectors" not in candidate.metadata:
                    candidate.metadata["source_detectors"] = []
                candidate.metadata["source_detectors"].append(detector.name)
                
                # Scale confidence by detector weight
                candidate.confidence *= self.weights[i]
                
            all_candidates.extend(detector_candidates)
            
        # Merge overlapping candidates
        merged_candidates = self._merge_overlapping_candidates(all_candidates)
        
        return merged_candidates
    
    def _merge_overlapping_candidates(self, 
                                     candidates: List[TransliterationCandidate]
                                     ) -> List[TransliterationCandidate]:
        """
        Merge overlapping candidates from different detectors
        
        Args:
            candidates: List of candidates from all detectors
            
        Returns:
            List of merged candidates
        """
        if not candidates:
            return []
            
        # Sort by start position
        sorted_candidates = sorted(candidates, key=lambda c: (c.start_pos, c.end_pos))
        
        merged = []
        current = sorted_candidates[0]
        
        for next_candidate in sorted_candidates[1:]:
            # Check if candidates overlap
            if next_candidate.start_pos <= current.end_pos:
                # Candidates overlap, merge them
                
                # Determine the span of the merged candidate
                end_pos = max(current.end_pos, next_candidate.end_pos)
                
                # If next candidate completely contains current, use next candidate's token
                if (next_candidate.start_pos <= current.start_pos and 
                    next_candidate.end_pos >= current.end_pos):
                    token = next_candidate.token
                # If current completely contains next, use current's token
                elif (current.start_pos <= next_candidate.start_pos and 
                      current.end_pos >= next_candidate.end_pos):
                    token = current.token
                # Otherwise, use the text span from the original text
                # This is a simplification - in practice, you'd need the original text
                else:
                    # Use the longer token as a fallback
                    token = current.token if len(current.token) > len(next_candidate.token) else next_candidate.token
                
                # Combine confidence scores (weighted average)
                total_weight = current.metadata.get("weight", 1.0) + next_candidate.metadata.get("weight", 1.0)
                combined_confidence = (
                    current.confidence * current.metadata.get("weight", 1.0) +
                    next_candidate.confidence * next_candidate.metadata.get("weight", 1.0)
                ) / total_weight
                
                # Merge metadata
                merged_metadata = current.metadata.copy()
                for key, value in next_candidate.metadata.items():
                    if key in merged_metadata and isinstance(value, list) and isinstance(merged_metadata[key], list):
                        # Combine lists without duplicates
                        merged_metadata[key] = list(set(merged_metadata[key] + value))
                    else:
                        merged_metadata[key] = value
                
                # Update current with merged information
                current = TransliterationCandidate(
                    token=token,
                    start_pos=current.start_pos,
                    end_pos=end_pos,
                    confidence=combined_confidence,
                    metadata=merged_metadata
                )
            else:
                # No overlap, add current to results and move to next
                merged.append(current)
                current = next_candidate
        
        # Add the last candidate
        merged.append(current)
        
        return merged
    
    def add_detector(self, detector: BaseCandidateDetector, weight: float = 1.0):
        """
        Add a new detector to the ensemble
        
        Args:
            detector: Detector instance to add
            weight: Weight for the new detector
        """
        self.detectors.append(detector)
        
        # Recalculate weights
        total_weight = sum(self.weights) + weight
        self.weights = [w * (sum(self.weights) / total_weight) for w in self.weights]
        self.weights.append(weight / total_weight)
        
        self.logger.info(f"Added detector {detector.name} with weight {weight}")
    
    def remove_detector(self, detector_name: str):
        """
        Remove a detector from the ensemble by name
        
        Args:
            detector_name: Name of the detector to remove
        """
        for i, detector in enumerate(self.detectors):
            if detector.name == detector_name:
                self.detectors.pop(i)
                removed_weight = self.weights.pop(i)
                
                # Renormalize weights if any remain
                if self.weights:
                    total = sum(self.weights)
                    self.weights = [w / total for w in self.weights]
                
                self.logger.info(f"Removed detector {detector_name}")
                return
                
        self.logger.warning(f"Detector {detector_name} not found in ensemble")