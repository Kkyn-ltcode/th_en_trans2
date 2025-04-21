from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from collections import defaultdict

from ..detection.base_detector import TransliterationCandidate
from .base_recovery import BaseRecoveryModule, TransliterationResult
from .context_reranker import ContextReranker

class EnsembleRecovery(BaseRecoveryModule):
    """
    Ensemble recovery module that combines multiple recovery approaches
    """
    def __init__(self, 
                 recovery_modules: List[BaseRecoveryModule],
                 weights: Optional[List[float]] = None,
                 use_context_reranking: bool = True,
                 context_reranker: Optional[ContextReranker] = None,
                 name: str = "ensemble_recovery"):
        """
        Initialize the ensemble recovery module
        
        Args:
            recovery_modules: List of recovery module instances
            weights: List of weights for each module (must match modules length)
            use_context_reranking: Whether to use context-based reranking
            context_reranker: Context reranker instance (created if None)
            name: Name of the module
        """
        super().__init__(name=name)
        
        self.recovery_modules = recovery_modules
        
        # Validate and normalize weights
        if weights is None:
            # Equal weights if not provided
            self.weights = [1.0 / len(recovery_modules)] * len(recovery_modules)
        else:
            if len(weights) != len(recovery_modules):
                raise ValueError("Number of weights must match number of recovery modules")
            
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        # Context reranking
        self.use_context_reranking = use_context_reranking
        if use_context_reranking:
            self.context_reranker = context_reranker or ContextReranker()
            
        self.logger.info(f"Initialized ensemble with {len(recovery_modules)} recovery modules")
    
    def recover(self, candidate: TransliterationCandidate) -> TransliterationResult:
        """
        Recover the original English form using ensemble of methods
        
        Args:
            candidate: The transliteration candidate to recover
            
        Returns:
            A TransliterationResult object with the recovered English form
        """
        # Get results from all recovery modules
        results = []
        for i, module in enumerate(self.recovery_modules):
            result = module.recover(candidate)
            
            # Apply module weight to confidence
            weighted_confidence = result.confidence * self.weights[i]
            
            # Store result with module info
            results.append((result, weighted_confidence, module.name))
        
        # Combine results
        return self._combine_results(results, candidate)
    
    def recover_batch(self, candidates: List[TransliterationCandidate]) -> List[TransliterationResult]:
        """
        Recover the original English forms for a batch of transliteration candidates
        
        Args:
            candidates: List of transliteration candidates to recover
            
        Returns:
            List of TransliterationResult objects
        """
        batch_results = []
        
        # Process each candidate
        for candidate in candidates:
            result = self.recover(candidate)
            batch_results.append(result)
            
        return batch_results
    
    def recover_with_context(self, 
                           candidate: TransliterationCandidate,
                           context_before: str,
                           context_after: str) -> TransliterationResult:
        """
        Recover with context information
        
        Args:
            candidate: The transliteration candidate to recover
            context_before: Text context before the word
            context_after: Text context after the word
            
        Returns:
            A TransliterationResult object with the recovered English form
        """
        # First get basic recovery result
        result = self.recover(candidate)
        
        # If we have candidates and context reranking is enabled
        if self.use_context_reranking and result.english_candidates:
            # Rerank candidates based on context
            reranked_candidates = self.context_reranker.rerank_candidates(
                result.english_candidates,
                context_before,
                context_after
            )
            
            if reranked_candidates:
                # Update result with reranked candidates
                best_candidate, best_score = reranked_candidates[0]
                
                # Create new result with reranked information
                result = TransliterationResult(
                    original_token=result.original_token,
                    english_candidates=[c for c, _ in reranked_candidates],
                    best_candidate=best_candidate,
                    confidence=best_score,
                    metadata={
                        **result.metadata,
                        "reranking": "context",
                        "original_best": result.best_candidate,
                        "original_confidence": result.confidence
                    }
                )
        
        return result
    
    def _combine_results(self, 
                        results: List[Tuple[TransliterationResult, float, str]],
                        candidate: TransliterationCandidate) -> TransliterationResult:
        """
        Combine results from multiple recovery modules
        
        Args:
            results: List of (result, weighted_confidence, module_name) tuples
            candidate: Original transliteration candidate
            
        Returns:
            Combined TransliterationResult
        """
        # If no results, return empty result
        if not results:
            return TransliterationResult(
                original_token=candidate.token,
                english_candidates=[],
                best_candidate="",
                confidence=0.0,
                metadata={"source": "ensemble_empty"}
            )
        
        # Collect all English candidates with their scores
        all_candidates = defaultdict(float)
        
        for result, weighted_confidence, module_name in results:
            # Skip empty results
            if not result.best_candidate:
                continue
                
            # Add best candidate with its weighted confidence
            all_candidates[result.best_candidate] += weighted_confidence
            
            # Add other candidates with reduced weight
            for i, cand in enumerate(result.english_candidates):
                if cand != result.best_candidate:
                    # Reduce weight for non-best candidates
                    reduction_factor = 0.8 ** (i + 1)
                    all_candidates[cand] += weighted_confidence * reduction_factor
        
        # Sort candidates by combined score
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
        
        # If no valid candidates, return empty result
        if not sorted_candidates:
            return TransliterationResult(
                original_token=candidate.token,
                english_candidates=[],
                best_candidate="",
                confidence=0.0,
                metadata={"source": "ensemble_no_valid_candidates"}
            )
        
        # Get best candidate and its confidence
        best_candidate, best_confidence = sorted_candidates[0]
        
        # Get all candidates
        english_candidates = [c for c, _ in sorted_candidates]
        
        # Create metadata with module contributions
        metadata = {
            "source": "ensemble",
            "module_contributions": {
                module_name: weighted_confidence
                for _, weighted_confidence, module_name in results
                if weighted_confidence > 0
            }
        }
        
        return TransliterationResult(
            original_token=candidate.token,
            english_candidates=english_candidates,
            best_candidate=best_candidate,
            confidence=best_confidence,
            metadata=metadata
        )