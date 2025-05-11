import os
import re
from utils.image_utils import encode_image_to_base64

EMOTION_LABELS_EN_AR = {
    'Joy': 'سعادة',
    'Trust': 'ثقة',
    'Fear': 'خوف',
    'Surprise': 'مفاجأة',
    'Sadness': 'حزن',
    'Disgust': 'قرف',
    'Anger': 'غضب',
    'Anticipation': 'ترقب',
    'neutral': 'محايد',
}

def normalize_emotion(text):
    """
    Normalize Arabic emotion text by removing diacritics, normalizing alefs,
    and matching to the closest emotion from the predefined list.
    
    Args:
        text: The emotion text to normalize
        
    Returns:
        Normalized emotion text that matches one of the predefined labels
    """
    if text is None:
        return None
        
    # Remove punctuation and extra spaces
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize alefs and other Arabic characters
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # Direct mapping for common variations
    variations = {
        # Common variations of سعادة (joy)
        'سعاده': 'سعادة',
        'فرح': 'سعادة',
        'فرحة': 'سعادة',
        'سرور': 'سعادة',
        'بهجة': 'سعادة',
        
        # Common variations of ثقة (trust)
        'ثقه': 'ثقة',
        'امان': 'ثقة',
        'اطمئنان': 'ثقة',
        
        # Common variations of خوف (fear)
        'رعب': 'خوف',
        'فزع': 'خوف',
        'خشية': 'خوف',
        
        # Common variations of مفاجأة (surprise)
        'مفاجاة': 'مفاجأة',
        'دهشة': 'مفاجأة',
        'ذهول': 'مفاجأة',
        
        # Common variations of حزن (sadness)
        'اسى': 'حزن',
        'كابة': 'حزن',
        'كآبة': 'حزن',
        'حسرة': 'حزن',
        
        # Common variations of قرف (disgust)
        'اشمئزاز': 'قرف',
        'استياء': 'قرف',
        
        # Common variations of غضب (anger)
        'سخط': 'غضب',
        'غيظ': 'غضب',
        
        # Common variations of ترقب (anticipation)
        'انتظار': 'ترقب',
        'توقع': 'ترقب',
        
        # Common variations of محايد (neutral)
        'حيادي': 'محايد',
        'محايدة': 'محايد',
    }
    
    if text in variations:
        return variations[text]
    
    # Check if the text is already in our predefined labels
    all_emotions = list(EMOTION_LABELS_EN_AR.values())
    if text in all_emotions:
        return text
    
    # Find closest match based on starting characters
    for emotion in all_emotions:
        if text.startswith(emotion[:2]):  # Match first 2 characters
            return emotion
    
    # No close match found, return as is
    return text

def get_zero_shot_prompt():
    return (
        "النظام: أنت خبير في علم النفس العاطفي للأطفال.\n"
        "المستخدم: انظر إلى الصورة التالية ثم أجب عن السؤال.\n\n"
        "[صورة: <IMAGE_PANEL>]\n\n"
        "السؤال: ما هو الشعور الأساسي الظاهر في هذا المشهد؟\n"
        "اختر كلمة واحدة فقط من القائمة التالية:\n"
        + '، '.join(EMOTION_LABELS_EN_AR.values()) + ".\n\n"
        "أجب بالكلمة المختارة فقط دون أي شرح إضافي."
    )

def get_few_shot_prompt(example_images=None):
    """
    Construct the few-shot prompt as a list of (text, image) tuples for the three examples, and a final (text, None) for the target prompt.
    Args:
        example_images: List of tuples (image_path, PIL.Image)
    Returns:
        List of (text, image) tuples for API construction.
    """
    if example_images is None or len(example_images) != 3:
        raise ValueError("You must provide exactly 3 example images for the few-shot prompt.")
    labels = ['حزن', 'مفاجأة', 'قرف']
    prompt_examples = []
    for idx, ((img_path, img), label) in enumerate(zip(example_images, labels), 1):
        text = f"مثال {idx}\nالسؤال: ما الشعور الأساسي؟\nالإجابة: {label}"
        prompt_examples.append((text, img))
    # Final target prompt (no image)
    target_text = (
        "الآن حلل الصورة الجديدة وأجب بالشعور الأساسي بكلمة واحدة فقط.\n"
        "السؤال: ما الشعور الأساسي؟\n"
        "اختر من: سعادة، ثقة، خوف، مفاجأة، حزن، قرف، غضب، ترقب، محايد."
    )
    prompt_examples.append((target_text, None))
    return prompt_examples

def get_chain_of_thought_prompt():
    return (
        "النظام: أنت خبير في علم النفس العاطفي للأطفال.\n"
        "المستخدم: انظر إلى هذه الصورة ثم أجب عن المطلوب.\n\n"
        "[صورة: <IMAGE_PANEL>]\n\n"
        "الخطوات:\n"
        "١) فكِّر خطوة بخطوة: صف بإيجاز تعابير الوجه أو لغة الجسد والعناصر السياقية التي تدل على الشعور (سطرين على الأكثر).\n"
        "٢) استنتج الشعور الأساسي الظاهر باستخدام كلمة واحدة فقط من القائمة: "
        + '، '.join(EMOTION_LABELS_EN_AR.values()) + ".\n"
        "٣) اطبع الإجابة النهائية في سطر منفصل بصيغة:\nالشعور: <الكلمة>\n\n"
        "ابدأ الآن."
    ) 