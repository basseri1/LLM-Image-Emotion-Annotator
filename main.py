import os
import logging
import csv
from datetime import datetime
from utils.image_utils import load_images
from utils.prompt_utils import (
    get_zero_shot_prompt,
    get_few_shot_prompt,
    get_chain_of_thought_prompt,
    EMOTION_LABELS_EN_AR,
    normalize_emotion
)
from utils.model_utils import (
    query_gpt4o,
    query_gemini
)
import time
from tqdm import tqdm

def load_named_few_shot_examples():
    """
    Load the three required few-shot example images by base name (sadness, surprise, disgust) regardless of extension.
    Returns a list of (path, PIL.Image) tuples in the order: sadness, surprise, disgust.
    """
    from PIL import Image
    few_shot_dir = 'few_shot_examples'
    required_basenames = [
        ('sadness', 'حزن'),
        ('surprise', 'مفاجأة'),
        ('disgust', 'قرف')
    ]
    supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp', '.ico', '.heic')
    files = os.listdir(few_shot_dir)
    loaded = []
    for base, label in required_basenames:
        found = None
        for f in files:
            if f.lower().startswith(base) and f.lower().endswith(supported_exts):
                found = f
                break
        if not found:
            raise FileNotFoundError(f"Few-shot example image for '{label}' not found: {base}.[image extension] in {few_shot_dir}")
        path = os.path.join(few_shot_dir, found)
        img = Image.open(path)
        loaded.append((path, img))
    return loaded

def get_temperature_from_user():
    """
    Prompts the user to select a temperature value for the models.
    
    Returns:
        float: The temperature value between 0 and 1
    """
    print("\n" + "="*80)
    print("MODEL TEMPERATURE SETTING")
    print("="*80)
    print("Temperature controls how deterministic the model responses are:")
    print("  • Lower temperature (0.0): More focused, consistent, and deterministic responses")
    print("  • Higher temperature (1.0): More creative, diverse, and exploratory responses")
    print()
    print("For emotion classification tasks:")
    print("  • Lower values (0.0-0.3): Better for consistent, accurate labeling")
    print("  • Higher values (0.4-0.7): May help when models are being too cautious/refusing")
    print("  • Very high values (0.8-1.0): Generally not recommended for classification tasks")
    print()
    
    while True:
        try:
            temp = input("Enter your desired temperature (0.0-1.0) [default: 0.0]: ").strip()
            if temp == "":
                return 0.0  # Default value
            
            temp = float(temp)
            if 0.0 <= temp <= 1.0:
                return temp
            else:
                print("Temperature must be between 0.0 and 1.0. Please try again.")
        except ValueError:
            print("Please enter a valid number between 0.0 and 1.0.")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get temperature setting from user
    temperature = get_temperature_from_user()
    print(f"Using temperature: {temperature}")
    
    images = load_images('images')
    total_images = len(images)
    print(f"Found {total_images} images in the 'images' folder.")
    
    few_shot_examples = load_named_few_shot_examples()
    results = []
    start_time = time.time()
    
    for idx, (img_path, img) in enumerate(tqdm(images, desc='Processing Images', unit='img'), 1):
        img_filename = os.path.basename(img_path)
        tqdm.write(f"\nProcessing image {idx}/{total_images}: {img_filename}")
        logging.info(f"Processing image {idx}/{total_images}: {img_path}")
        
        row = {
            'image_name': img_filename,
            'gpt4o_zero_shot': None,
            'gpt4o_few_shot': None,
            'gpt4o_cot': None,
            'gpt4o_cot_reasoning': None,
            'gemini_zero_shot': None,
            'gemini_few_shot': None,
            'gemini_cot': None,
            'gemini_cot_reasoning': None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        # GPT-4o
        result = query_gpt4o(img, None, 'zero_shot', temperature=temperature)
        normalized_label = normalize_emotion(result['label'])
        row['gpt4o_zero_shot'] = normalized_label
        print(f"Image {idx}/{total_images}: {img_filename}\nModel: gpt-4o\nPrompt type: zero_shot\nLabel: {result['label']}")
        if normalized_label != result['label']:
            print(f"Normalized to: {normalized_label}")
        print('-' * 40)
        
        result = query_gpt4o(img, None, 'few_shot', temperature=temperature, few_shot_examples=few_shot_examples)
        normalized_label = normalize_emotion(result['label'])
        row['gpt4o_few_shot'] = normalized_label
        print(f"Image {idx}/{total_images}: {img_filename}\nModel: gpt-4o\nPrompt type: few_shot\nLabel: {result['label']}")
        if normalized_label != result['label']:
            print(f"Normalized to: {normalized_label}")
        print('-' * 40)
        
        result = query_gpt4o(img, None, 'chain_of_thought', temperature=temperature)
        normalized_label = normalize_emotion(result['label'])
        row['gpt4o_cot'] = normalized_label
        row['gpt4o_cot_reasoning'] = result.get('reasoning')
        print(f"Image {idx}/{total_images}: {img_filename}\nModel: gpt-4o\nPrompt type: chain_of_thought\nLabel: {result['label']}")
        if normalized_label != result['label']:
            print(f"Normalized to: {normalized_label}")
        if result.get('reasoning'):
            print(f"Reasoning: {result['reasoning']}")
        print('-' * 40)
        
        # Gemini
        result = query_gemini(img, None, 'zero_shot', temperature=temperature)
        normalized_label = normalize_emotion(result['label'])
        row['gemini_zero_shot'] = normalized_label
        print(f"Image {idx}/{total_images}: {img_filename}\nModel: gemini-1.5-pro\nPrompt type: zero_shot\nLabel: {result['label']}")
        if normalized_label != result['label']:
            print(f"Normalized to: {normalized_label}")
        print('-' * 40)
        
        result = query_gemini(img, None, 'few_shot', temperature=temperature, few_shot_examples=few_shot_examples)
        normalized_label = normalize_emotion(result['label'])
        row['gemini_few_shot'] = normalized_label
        print(f"Image {idx}/{total_images}: {img_filename}\nModel: gemini-1.5-pro\nPrompt type: few_shot\nLabel: {result['label']}")
        if normalized_label != result['label']:
            print(f"Normalized to: {normalized_label}")
        print('-' * 40)
        
        result = query_gemini(img, None, 'chain_of_thought', temperature=temperature)
        normalized_label = normalize_emotion(result['label'])
        row['gemini_cot'] = normalized_label
        row['gemini_cot_reasoning'] = result.get('reasoning')
        print(f"Image {idx}/{total_images}: {img_filename}\nModel: gemini-1.5-pro\nPrompt type: chain_of_thought\nLabel: {result['label']}")
        if normalized_label != result['label']:
            print(f"Normalized to: {normalized_label}")
        if result.get('reasoning'):
            print(f"Reasoning: {result['reasoning']}")
        print('-' * 40)
        
        results.append(row)

        # Elapsed and remaining time reporting
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (total_images - idx)
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        remaining_str = time.strftime('%H:%M:%S', time.gmtime(remaining))
        tqdm.write(f"Elapsed time: {elapsed_str} | Estimated remaining: {remaining_str}")
    # Save results to CSV
    os.makedirs('results', exist_ok=True)
    csv_filename = f"results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'image_name',
            'gpt4o_zero_shot',
            'gpt4o_few_shot',
            'gpt4o_cot',
            'gpt4o_cot_reasoning',
            'gemini_zero_shot',
            'gemini_few_shot',
            'gemini_cot',
            'gemini_cot_reasoning',
            'timestamp'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    logging.info(f"Results saved to {csv_filename}")
    logging.info("Processing complete.")

if __name__ == '__main__':
    main() 