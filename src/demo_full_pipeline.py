import os
import sys
import logging
from typing import List, Dict, Any

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

# Import the pipeline
from src.pipeline.transliteration_pipeline import TransliterationPipeline

def create_sample_dictionary():
    """Create a sample Thai-English transliteration dictionary"""
    # Create a temporary dictionary file
    dict_path = os.path.join(project_root, "data", "sample_dictionary.json")
    os.makedirs(os.path.dirname(dict_path), exist_ok=True)
    
    import json
    sample_dict = {
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
        "ไฟล์": "file",
        "เฟซบุ๊ก": "Facebook",
        "ทวิตเตอร์": "Twitter",
        "อินสตาแกรม": "Instagram",
        "ยูทูบ": "YouTube",
        "กูเกิล": "Google"
    }
    
    with open(dict_path, 'w', encoding='utf-8') as f:
        json.dump(sample_dict, f, ensure_ascii=False, indent=2)
    
    return dict_path

def print_results(original: str, processed: str, info: List[Dict[str, Any]]):
    """Print the results in a formatted way"""
    print("\nOriginal Thai text:")
    print(f"  {original}")
    
    print("\nProcessed text:")
    print(f"  {processed}")
    
    if info:
        print("\nDetected transliterations:")
        for i, item in enumerate(info):
            print(f"  {i+1}. '{item['thai']}' → '{item['english']}' (confidence: {item['confidence']:.2f})")
            if len(item['alternatives']) > 1:
                print(f"     Alternatives: {', '.join(item['alternatives'][1:5])}")
    else:
        print("\nNo transliterations detected")
    
    print("\n" + "-"*50)

def main():
    # Create a sample dictionary
    dict_path = create_sample_dictionary()
    
    # Initialize the pipeline
    pipeline = TransliterationPipeline(
        detector_type="ensemble",
        recovery_type="ensemble",
        dictionary_path=dict_path,
        use_context=True
    )
    
    # Example Thai sentences with transliterations
    example_texts = [
        "ฉันใช้คอมพิวเตอร์ทุกวันเพื่อเช็คอีเมล",  # "I use a computer every day to check email"
        "เขาชอบเล่นเกมออนไลน์บนโทรศัพท์มือถือ",  # "He likes to play online games on his mobile phone"
        "ฉันดาวน์โหลดแอพพลิเคชั่นใหม่จากแอปสโตร์",  # "I downloaded a new application from the App Store"
        "เธอแชร์วิดีโอบนเฟซบุ๊กและอินสตาแกรม",  # "She shared a video on Facebook and Instagram"
        "บริษัทกำลังพัฒนาซอฟต์แวร์ใหม่สำหรับระบบคลาวด์",  # "The company is developing new software for cloud systems"
    ]
    
    # Process each example
    for i, text in enumerate(example_texts):
        print(f"\n\n{'='*50}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*50}")
        
        # Process the text
        processed_text, info = pipeline.process_text(text)
        print_results(text, processed_text, info)
        
        # Show with markup
        marked_text = pipeline.process_text_with_markup(text, markup_format="console")
        print("Text with console markup:")
        print(f"  {marked_text}")
        
        # Show with custom replacement
        replaced_text = pipeline.process_text_with_replacement(text, replacement_format="[{english}]")
        print("\nText with custom replacement:")
        print(f"  {replaced_text}")

if __name__ == "__main__":
    main()