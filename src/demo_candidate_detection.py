import os
import sys
import logging
from typing import List

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
from src.detection.ngram_detector import NgramDetector
from src.detection.transformer_detector import TransformerDetector
from src.detection.ensemble_detector import EnsembleDetector

def print_candidates(candidates: List[TransliterationCandidate], source: str):
    """Print detected candidates in a formatted way"""
    print(f"\n=== Candidates detected by {source} ===")
    if not candidates:
        print("No candidates detected")
        return
        
    for i, candidate in enumerate(candidates):
        print(f"{i+1}. '{candidate.token}' (confidence: {candidate.confidence:.2f})")
        print(f"   Position: {candidate.start_pos}-{candidate.end_pos}")
        if "source_detectors" in candidate.metadata:
            print(f"   Sources: {', '.join(candidate.metadata['source_detectors'])}")
        if "rules_matched" in candidate.metadata:
            print(f"   Rules matched: {', '.join(candidate.metadata['rules_matched'])}")
        print()

def demo_rule_based(text: str):
    """Demo rule-based detection"""
    tokenizer = ThaiTokenizer()
    detector = RuleBasedDetector(tokenizer=tokenizer)
    candidates = detector.detect_candidates(text)
    print_candidates(candidates, "Rule-Based Detector")
    return detector, candidates

def demo_ngram(text: str, train_data: bool = False):
    """Demo n-gram based detection"""
    tokenizer = ThaiTokenizer()
    detector = NgramDetector(tokenizer=tokenizer)
    
    # For demo purposes, we'll use a very small training set
    # In a real scenario, you would use a much larger dataset
    if train_data:
        # Example training data
        thai_texts = [
            "วันนี้อากาศดีมาก",
            "ฉันชอบกินอาหารไทย",
            "ประเทศไทยมีประชากรมาก"
        ]
        
        trans_texts = [
            "ฉันใช้คอมพิวเตอร์ทุกวัน",
            "เขาชอบเล่นเกมออนไลน์",
            "ฉันส่งอีเมลไปหาเพื่อน"
        ]
        
        detector.train(thai_texts, trans_texts)
        logger.info("Trained n-gram model with demo data")
    else:
        logger.warning("Using untrained n-gram model - results will be random")
    
    candidates = detector.detect_candidates(text)
    print_candidates(candidates, "N-gram Detector")
    return detector, candidates

def demo_transformer(text: str, model_available: bool = False):
    """Demo transformer-based detection"""
    tokenizer = ThaiTokenizer()
    
    if model_available:
        # In a real scenario, you would have a fine-tuned model
        model_path = "/Users/nguyen/Documents/Work/Thai_English_Transliteration1/models/transliteration_detector"
        detector = TransformerDetector(model_name_or_path=model_path, thai_tokenizer=tokenizer)
    else:
        # For demo, we'll use a base model but note it's not fine-tuned
        detector = TransformerDetector(thai_tokenizer=tokenizer)
        logger.warning("Using base transformer model without fine-tuning - results will be unreliable")
    
    # For demo purposes, we'll simulate detection results
    # In a real scenario, this would use the actual model predictions
    if not model_available:
        # Return empty results since we don't have a trained model
        print("\n=== Candidates detected by Transformer Detector ===")
        print("No candidates detected (model not fine-tuned)")
        return detector, []
    
    candidates = detector.detect_candidates(text)
    print_candidates(candidates, "Transformer Detector")
    return detector, candidates

def demo_ensemble(text: str, detectors: List):
    """Demo ensemble detection"""
    # Create ensemble with the detectors
    ensemble = EnsembleDetector(
        detectors=detectors,
        weights=[0.4, 0.6]  # Giving more weight to the second detector
    )
    
    candidates = ensemble.detect_candidates(text)
    print_candidates(candidates, "Ensemble Detector")
    return ensemble, candidates

def main():
    # Example Thai text with transliterations
    example_texts = [
        "ฉันใช้คอมพิวเตอร์ทุกวันเพื่อเช็คอีเมล",  # "I use a computer every day to check email"
        "เขาชอบเล่นเกมออนไลน์บนโทรศัพท์มือถือ",  # "He likes to play online games on his mobile phone"
        "ฉันดาวน์โหลดแอพพลิเคชั่นใหม่จากแอปสโตร์",  # "I downloaded a new application from the App Store"
    ]
    
    for i, text in enumerate(example_texts):
        print(f"\n\n{'='*50}")
        print(f"EXAMPLE {i+1}: {text}")
        print(f"{'='*50}")
        
        # Run individual detectors
        rule_detector, rule_candidates = demo_rule_based(text)
        ngram_detector, ngram_candidates = demo_ngram(text, train_data=True)
        
        # Skip transformer demo since we don't have a fine-tuned model
        # transformer_detector, transformer_candidates = demo_transformer(text)
        
        # Run ensemble with available detectors
        ensemble, ensemble_candidates = demo_ensemble(
            text, 
            [rule_detector, ngram_detector]
        )
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()