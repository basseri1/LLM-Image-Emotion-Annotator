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
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text
import arabic_reshaper
from bidi.algorithm import get_display

console = Console()

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
    console.print(Panel("[bold yellow]MODEL TEMPERATURE SETTING[/bold yellow]", style="bold blue"))
    rprint("[bold]Temperature controls how deterministic the model responses are:[/bold]")
    rprint("  • [green]Lower temperature (0.0):[/green] More focused, consistent, and deterministic responses")
    rprint("  • [magenta]Higher temperature (1.0):[/magenta] More creative, diverse, and exploratory responses\n")
    rprint("[bold]For emotion classification tasks:[/bold]")
    rprint("  • [green]Lower values (0.0-0.3):[/green] Better for consistent, accurate labeling")
    rprint("  • [yellow]Higher values (0.4-0.7):[/yellow] May help when models are being too cautious/refusing")
    rprint("  • [red]Very high values (0.8-1.0):[/red] Generally not recommended for classification tasks\n")
    while True:
        try:
            temp = Prompt.ask("[bold cyan]Enter your desired temperature (0.0-1.0)[/bold cyan]", default="0.0")
            temp = temp.strip()
            if temp == "":
                return 0.0  # Default value
            
            temp = float(temp)
            if 0.0 <= temp <= 1.0:
                return temp
            else:
                rprint("[red]Temperature must be between 0.0 and 1.0. Please try again.[/red]")
        except ValueError:
            rprint("[red]Please enter a valid number between 0.0 and 1.0.[/red]")

def reshape_arabic(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

def print_tqdm_rich(text):
    """
    Helper function to print rich-formatted text with tqdm
    by capturing console output and then writing plain text to tqdm.
    """
    # Use console.capture() to grab rich output
    with console.capture() as capture:
        console.print(text)
    tqdm.write(capture.get())

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get temperature setting from user
    temperature = get_temperature_from_user()
    rprint(f":thermometer: [bold cyan]Using temperature:[/bold cyan] [yellow]{temperature}[/yellow]")
    
    images = load_images('images')
    total_images = len(images)
    rprint(f":framed_picture: [bold green]Found {total_images} images in the 'images' folder.[/bold green]")
    
    few_shot_examples = load_named_few_shot_examples()
    results = []
    start_time = time.time()
    
    for idx, (img_path, img) in enumerate(tqdm(images, desc='Processing Images', unit='img'), 1):
        img_filename = os.path.basename(img_path)
        print_tqdm_rich(f"\n[bold blue]Processing image {idx}/{total_images}: {img_filename}[/bold blue]")
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
            'gpt4o_zero_shot_request': None,
            'gpt4o_few_shot_request': None,
            'gpt4o_cot_request': None,
            'gemini_zero_shot_request': None,
            'gemini_few_shot_request': None,
            'gemini_cot_request': None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        # GPT-4o
        result = query_gpt4o(img, None, 'zero_shot', temperature=temperature)
        normalized_label = normalize_emotion(result['label'])
        arabic_label = reshape_arabic(normalized_label)
        row['gpt4o_zero_shot'] = normalized_label
        row['gpt4o_zero_shot_request'] = result.get('request_json')
        rprint(Panel(f"[bold]Image {idx}/{total_images}: [cyan]{img_filename}[/cyan]\nModel: [magenta]gpt-4o[/magenta]\nPrompt type: [yellow]zero_shot[/yellow]\nLabel: [green]{arabic_label}[/green]", title=":robot: GPT-4o Zero-Shot", style="bold blue"))
        if normalized_label != result['label']:
            arabic_normalized_label = reshape_arabic(normalized_label)
            rprint(f"[yellow]Normalized to:[/yellow] [green]{arabic_normalized_label}[/green]")
        rprint('-' * 40)
        
        result = query_gpt4o(img, None, 'few_shot', temperature=temperature, few_shot_examples=few_shot_examples)
        normalized_label = normalize_emotion(result['label'])
        arabic_label = reshape_arabic(normalized_label)
        row['gpt4o_few_shot'] = normalized_label
        row['gpt4o_few_shot_request'] = result.get('request_json')
        rprint(Panel(f"[bold]Image {idx}/{total_images}: [cyan]{img_filename}[/cyan]\nModel: [magenta]gpt-4o[/magenta]\nPrompt type: [yellow]few_shot[/yellow]\nLabel: [green]{arabic_label}[/green]", title=":robot: GPT-4o Few-Shot", style="bold magenta"))
        if normalized_label != result['label']:
            arabic_normalized_label = reshape_arabic(normalized_label)
            rprint(f"[yellow]Normalized to:[/yellow] [green]{arabic_normalized_label}[/green]")
        rprint('-' * 40)
        
        result = query_gpt4o(img, None, 'chain_of_thought', temperature=temperature)
        normalized_label = normalize_emotion(result['label'])
        arabic_label = reshape_arabic(normalized_label)
        row['gpt4o_cot'] = normalized_label
        row['gpt4o_cot_reasoning'] = result.get('reasoning')
        row['gpt4o_cot_request'] = result.get('request_json')
        rprint(Panel(f"[bold]Image {idx}/{total_images}: [cyan]{img_filename}[/cyan]\nModel: [magenta]gpt-4o[/magenta]\nPrompt type: [yellow]chain_of_thought[/yellow]\nLabel: [green]{arabic_label}[/green]", title=":robot: GPT-4o CoT", style="bold green"))
        if normalized_label != result['label']:
            arabic_normalized_label = reshape_arabic(normalized_label)
            rprint(f"[yellow]Normalized to:[/yellow] [green]{arabic_normalized_label}[/green]")
        if result.get('reasoning'):
            arabic_reasoning = reshape_arabic(result['reasoning'])
            rprint(f"[bold blue]Reasoning:[/bold blue] {arabic_reasoning}")
        rprint('-' * 40)
        
        # Gemini
        result = query_gemini(img, None, 'zero_shot', temperature=temperature)
        normalized_label = normalize_emotion(result['label'])
        arabic_label = reshape_arabic(normalized_label)
        row['gemini_zero_shot'] = normalized_label
        row['gemini_zero_shot_request'] = result.get('request_json')
        rprint(Panel(f"[bold]Image {idx}/{total_images}: [cyan]{img_filename}[/cyan]\nModel: [magenta]gemini-1.5-pro[/magenta]\nPrompt type: [yellow]zero_shot[/yellow]\nLabel: [green]{arabic_label}[/green]", title=":crystal_ball: Gemini Zero-Shot", style="bold blue"))
        if normalized_label != result['label']:
            arabic_normalized_label = reshape_arabic(normalized_label)
            rprint(f"[yellow]Normalized to:[/yellow] [green]{arabic_normalized_label}[/green]")
        rprint('-' * 40)
        
        result = query_gemini(img, None, 'few_shot', temperature=temperature, few_shot_examples=few_shot_examples)
        normalized_label = normalize_emotion(result['label'])
        arabic_label = reshape_arabic(normalized_label)
        row['gemini_few_shot'] = normalized_label
        row['gemini_few_shot_request'] = result.get('request_json')
        rprint(Panel(f"[bold]Image {idx}/{total_images}: [cyan]{img_filename}[/cyan]\nModel: [magenta]gemini-1.5-pro[/magenta]\nPrompt type: [yellow]few_shot[/yellow]\nLabel: [green]{arabic_label}[/green]", title=":crystal_ball: Gemini Few-Shot", style="bold magenta"))
        if normalized_label != result['label']:
            arabic_normalized_label = reshape_arabic(normalized_label)
            rprint(f"[yellow]Normalized to:[/yellow] [green]{arabic_normalized_label}[/green]")
        rprint('-' * 40)
        
        result = query_gemini(img, None, 'chain_of_thought', temperature=temperature)
        normalized_label = normalize_emotion(result['label'])
        arabic_label = reshape_arabic(normalized_label)
        row['gemini_cot'] = normalized_label
        row['gemini_cot_reasoning'] = result.get('reasoning')
        row['gemini_cot_request'] = result.get('request_json')
        rprint(Panel(f"[bold]Image {idx}/{total_images}: [cyan]{img_filename}[/cyan]\nModel: [magenta]gemini-1.5-pro[/magenta]\nPrompt type: [yellow]chain_of_thought[/yellow]\nLabel: [green]{arabic_label}[/green]", title=":crystal_ball: Gemini CoT", style="bold green"))
        if normalized_label != result['label']:
            arabic_normalized_label = reshape_arabic(normalized_label)
            rprint(f"[yellow]Normalized to:[/yellow] [green]{arabic_normalized_label}[/green]")
        if result.get('reasoning'):
            arabic_reasoning = reshape_arabic(result['reasoning'])
            rprint(f"[bold blue]Reasoning:[/bold blue] {arabic_reasoning}")
        rprint('-' * 40)
        
        results.append(row)

        # Elapsed and remaining time reporting
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (total_images - idx)
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        remaining_str = time.strftime('%H:%M:%S', time.gmtime(remaining))
        print_tqdm_rich(f"[bold green]Elapsed time:[/bold green] {elapsed_str} | [bold yellow]Estimated remaining:[/bold yellow] {remaining_str}")
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
            'gpt4o_zero_shot_request',
            'gpt4o_few_shot_request',
            'gpt4o_cot_request',
            'gemini_zero_shot_request',
            'gemini_few_shot_request',
            'gemini_cot_request',
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