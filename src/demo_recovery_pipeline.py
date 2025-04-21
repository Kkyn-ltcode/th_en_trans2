import os
import sys
import logging
from typing import List, Dict, Tuple

# Add the project root to the Python path to ensure imports work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo")

# Import detection modules
from src.utils.tokenizer import ThaiTokenizer
from src.detection.base_detector import TransliterationCandidate
from src.detection.rule_based_detector import RuleBasedDetector

# Import recovery modules
from src.recovery.base_recovery import TransliterationResult, BaseRecoveryModule
from src.recovery.dictionary_recovery import DictionaryRecovery
from src.recovery.g2p_module import ThaiG2P
from src.recovery.phoneme_mapper import PhonemeMapper
from src.recovery.neural_recovery import NeuralRecovery
from src.recovery.ensemble_recovery import EnsembleRecovery
from src.recovery.context_reranker import ContextReranker

def print_recovery_result(result: TransliterationResult, source: str):
    """Print recovery result in a formatted way"""
    print(f"\n=== Recovery result from {source} ===")
    print(f"Original Thai token: '{result.original_token}'")
    print(f"Best English candidate: '{result.best_candidate}' (confidence: {result.confidence:.2f})")
    
    if result.english_candidates:
        print("All candidates:")
        for i, candidate in enumerate(result.english_candidates[:5]):  # Show top 5
            print(f"  {i+1}. '{candidate}'")
        
        if len(result.english_candidates) > 5:
            print(f"  ... and {len(result.english_candidates) - 5} more")
    else:
        print("No English candidates found")
    
    if result.metadata:
        print("Metadata:")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")

def create_sample_dictionary():
    """Create a sample Thai-English transliteration dictionary"""
    return {
        "คอมพิวเตอร์": "computer",
        "อินเทอร์เน็ต": "internet",
        "แอปพลิเคชัน": "application",
        "โทรศัพท์": "telephone",
        "เทคโนโลยี": "technology",
        "ซอฟต์แวร์": "software",
        "ฮาร์ดแวร์": "hardware",
        "เกม": "game",
        "ออนไลน์": "online",
        "อีเมล": "email",
        "เว็บไซต์": "website",
        "ดาวน์โหลด": "download",
        "อัปโหลด": "upload",
        "วิดีโอ": "video",
        "ไฟล์": "file"
    }

def demo_dictionary_recovery(candidate: TransliterationCandidate):
    """Demo dictionary-based recovery"""
    # Create a dictionary recovery module with sample dictionary
    sample_dict = create_sample_dictionary()
    recovery = DictionaryRecovery(name="sample_dictionary")
    
    # Manually set the dictionary
    recovery.thai_to_english = sample_dict
    for thai, english in sample_dict.items():
        recovery.english_to_thai[english].append(thai)
    
    # Recover the candidate
    result = recovery.recover(candidate)
    print_recovery_result(result, "Dictionary Recovery")
    return recovery, result

def demo_neural_recovery(candidate: TransliterationCandidate, use_real_model: bool = False):
    """Demo neural recovery"""
    if use_real_model:
        # Use a real model if available
        recovery = NeuralRecovery(model_name_or_path="google/mt5-small")
        result = recovery.recover(candidate)
    else:
        # Create a simple implementation of BaseRecoveryModule for simulation
        class SimulatedRecovery(BaseRecoveryModule):
            def recover(self, candidate):
                # Create a simulated result
                english_candidates = []
                if candidate.token in create_sample_dictionary():
                    # If in our sample dictionary, use that
                    english = create_sample_dictionary()[candidate.token]
                    english_candidates = [english, f"{english}s", f"{english}ing"]
                else:
                    # Otherwise make up some candidates
                    english_candidates = [
                        candidate.token.replace("ค", "c").replace("อ", "o"),
                        candidate.token.replace("ค", "k").replace("อ", "o"),
                        candidate.token[::-1]  # Reversed as a placeholder
                    ]
                
                return TransliterationResult(
                    original_token=candidate.token,
                    english_candidates=english_candidates,
                    best_candidate=english_candidates[0] if english_candidates else "",
                    confidence=0.85,
                    metadata={"source": "simulated_neural_model"}
                )
                
            def recover_batch(self, candidates):
                return [self.recover(c) for c in candidates]
        
        # Create an instance of our simulated recovery
        recovery = SimulatedRecovery(name="simulated_neural")
        result = recovery.recover(candidate)
        
        print("\n=== Neural Recovery (SIMULATED) ===")
        print("Note: Using simulated results since no real model is loaded")
    
    print_recovery_result(result, "Neural Recovery")
    return recovery, result

def demo_ensemble_recovery(candidate: TransliterationCandidate, recovery_modules: List):
    """Demo ensemble recovery"""
    # Create ensemble with the recovery modules
    ensemble = EnsembleRecovery(
        recovery_modules=recovery_modules,
        weights=[0.6, 0.4],  # Giving more weight to the first module
        use_context_reranking=True
    )
    
    # Recover without context
    result = ensemble.recover(candidate)
    print_recovery_result(result, "Ensemble Recovery (without context)")
    
    # Recover with context
    context_before = "I use my"
    context_after = "to browse the internet"
    
    context_result = ensemble.recover_with_context(
        candidate,
        context_before,
        context_after
    )
    print_recovery_result(context_result, "Ensemble Recovery (with context)")
    
    return ensemble, result

def main():
    # Create a sample transliteration candidate
    sample_candidates = [
        TransliterationCandidate(
            token="คอมพิวเตอร์",
            start_pos=0,
            end_pos=11,
            confidence=0.95,
            metadata={"source": "rule_based"}
        ),
        TransliterationCandidate(
            token="อินเทอร์เน็ต",
            start_pos=0,
            end_pos=11,
            confidence=0.92,
            metadata={"source": "rule_based"}
        ),
        TransliterationCandidate(
            token="เทคโนโลยี",
            start_pos=0,
            end_pos=9,
            confidence=0.88,
            metadata={"source": "rule_based"}
        )
    ]
    
    for i, candidate in enumerate(sample_candidates):
        print(f"\n\n{'='*50}")
        print(f"EXAMPLE {i+1}: {candidate.token}")
        print(f"{'='*50}")
        
        # Run individual recovery modules
        dict_recovery, dict_result = demo_dictionary_recovery(candidate)
        neural_recovery, neural_result = demo_neural_recovery(candidate)
        
        # Run ensemble recovery
        ensemble, ensemble_result = demo_ensemble_recovery(
            candidate, 
            [dict_recovery, neural_recovery]
        )
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()