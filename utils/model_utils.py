import os
from PIL import Image
from utils.image_utils import encode_image_to_base64
import openai
import google.generativeai as genai
from dotenv import load_dotenv
import io
import json

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

openai.api_key = OPENAI_API_KEY

def is_refusal_message(text):
    """
    Check if the response is a refusal message in Arabic.
    Returns True if it seems to be a refusal, False otherwise.
    """
    if text is None:
        return False
        
    # Common refusal patterns in Arabic
    refusal_patterns = [
        'آسف', 'عذرا', 'لا أستطيع', 'لا يمكنني', 'عفوا',
        'تحليل الصور', 'وصف الصور', 'التعرف على',
        'الأشخاص', 'غير قادر', 'المساعدة'
    ]
    
    # Count how many patterns match
    match_count = sum(1 for pattern in refusal_patterns if pattern in text)
    
    # If multiple patterns match, it's likely a refusal
    return match_count >= 2

def build_gpt4o_zero_shot_message(image):
    img_b64 = encode_image_to_base64(image, format='PNG')
    return [
        {"role": "system", "content": "أنت خبير في علم النفس العاطفي للأطفال."},
        {"role": "user", "content": [
            {"type": "text", "text": "انظر إلى الصورة التالية ثم أجب عن السؤال."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": "السؤال: ما هو الشعور الأساسي الظاهر في هذا المشهد؟\nاختر كلمة واحدة فقط من القائمة التالية :\nسعادة، ثقة، خوف، مفاجأة، حزن، قرف، غضب، ترقب، محايد.\nأجب بالكلمة المختارة فقط دون أي شرح إضافي."}
        ]}
    ]

def build_gpt4o_few_shot_message(image, few_shot_examples):
    """
    Constructs the few-shot prompt for GPT-4o in a fully manual, explicit order.
    
    IMPORTANT:
    - The order and formatting of few-shot examples and the target image/question are controlled explicitly.
    - Each example is introduced, followed by its image, then its answer, and then the next example.
    - After all examples, the target image and question are added with a clear transition.
    - This approach avoids ambiguity and ensures the model always understands the task, reducing refusals.
    - Even small changes in order, grouping, or newlines can cause refusals or unreliable answers from vision models.
    """
    user_content = []
    # Example 1: حزن
    user_content.append({"type": "text", "text": "أمثلة توضيحية:\nمثال ١"})
    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(few_shot_examples[0][1], format='PNG')}"}})
    user_content.append({"type": "text", "text": "السؤال: ما الشعور الأساسي؟\nالإجابة: حزن\nمثال ٢"})
    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(few_shot_examples[1][1], format='PNG')}"}})
    user_content.append({"type": "text", "text": "السؤال: ما الشعور الأساسي؟\nالإجابة: مفاجأة\nمثال ٣"})
    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(few_shot_examples[2][1], format='PNG')}"}})
    user_content.append({"type": "text", "text": "السؤال: ما الشعور الأساسي؟\nالإجابة: قرف\nالآن حلل الصورة الجديدة وأجب بالشعور الأساسي بكلمة واحدة فقط."})
    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(image, format='PNG')}"}})
    user_content.append({"type": "text", "text": "السؤال: ما الشعور الأساسي؟\nاختر من: سعادة، ثقة، خوف، مفاجأة، حزن، قرف، غضب، ترقب، محايد."})
    return [
        {"role": "system", "content": "أنت خبير في علم النفس العاطفي للأطفال."},
        {"role": "user", "content": user_content}
    ]

def build_gpt4o_cot_message(image):
    img_b64 = encode_image_to_base64(image, format='PNG')
    return [
        {"role": "system", "content": "أنت خبير في علم النفس العاطفي للأطفال."},
        {"role": "user", "content": [
            {"type": "text", "text": "انظر إلى هذه الصورة ثم أجب عن المطلوب."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": "الخطوات:\n١( فكِّر خطوة بخطوة: صف بإيجاز تعابير الوجه أو لغة الجسد والعناصر السياقية التي تدل على الشعور )سطرين على الأكثر(.\n٢( استنتج الشعور الأساسي الظاهر باستخدام كلمة واحدة فقط من القائمة:\nسعادة، ثقة، خوف، مفاجأة، حزن، قرف، غضب، ترقب، محايد.\n٣( اطبع الإجابة النهائية في سطر منفصل بصيغة:\nالشعور: >الكلمة<\nابدأ الآن."}
        ]}
    ]

def build_gemini_zero_shot_content(image):
    def image_to_binary(img):
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    img_binary = image_to_binary(image)
    system_prompt = "أنت خبير في علم النفس العاطفي للأطفال."
    user_prompt = "انظر إلى الصورة التالية ثم أجب عن السؤال.\nالسؤال: ما هو الشعور الأساسي الظاهر في هذا المشهد؟\nاختر كلمة واحدة فقط من القائمة التالية :\nسعادة، ثقة، خوف، مفاجأة، حزن، قرف، غضب، ترقب، محايد.\nأجب بالكلمة المختارة فقط دون أي شرح إضافي."
    return [
        {"role": "user", "parts": [
            {"text": system_prompt + "\n\n" + user_prompt},
            {"inline_data": {"mime_type": "image/png", "data": img_binary}}
        ]}
    ]

def build_gemini_few_shot_content(image, few_shot_examples):
    """
    Constructs the few-shot prompt for Gemini in a fully manual, explicit order.
    
    IMPORTANT:
    - The order and formatting of few-shot examples and the target image/question are controlled explicitly.
    - Each example is introduced, followed by its image, then its answer, and then the next example.
    - After all examples, the target image and question are added with a clear transition.
    - This approach avoids ambiguity and ensures the model always understands the task, reducing refusals.
    - Even small changes in order, grouping, or newlines can cause refusals or unreliable answers from vision models.
    """
    def image_to_binary(img):
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    content_parts = []
    # Example 1: حزن
    content_parts.append({"text": "أمثلة توضيحية:\nمثال ١"})
    content_parts.append({"inline_data": {"mime_type": "image/png", "data": image_to_binary(few_shot_examples[0][1])}})
    content_parts.append({"text": "السؤال: ما الشعور الأساسي؟\nالإجابة: حزن\nمثال ٢"})
    content_parts.append({"inline_data": {"mime_type": "image/png", "data": image_to_binary(few_shot_examples[1][1])}})
    content_parts.append({"text": "السؤال: ما الشعور الأساسي؟\nالإجابة: مفاجأة\nمثال ٣"})
    content_parts.append({"inline_data": {"mime_type": "image/png", "data": image_to_binary(few_shot_examples[2][1])}})
    content_parts.append({"text": "السؤال: ما الشعور الأساسي؟\nالإجابة: قرف\nالآن حلل الصورة الجديدة وأجب بالشعور الأساسي بكلمة واحدة فقط."})
    content_parts.append({"inline_data": {"mime_type": "image/png", "data": image_to_binary(image)}})
    content_parts.append({"text": "السؤال: ما الشعور الأساسي؟\nاختر من: سعادة، ثقة، خوف، مفاجأة، حزن، قرف، غضب، ترقب، محايد."})
    return [
        {"role": "user", "parts": content_parts}
    ]

def build_gemini_cot_content(image):
    def image_to_binary(img):
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    img_binary = image_to_binary(image)
    system_prompt = "أنت خبير في علم النفس العاطفي للأطفال."
    user_prompt = "انظر إلى هذه الصورة ثم أجب عن المطلوب.\nالخطوات:\n١( فكِّر خطوة بخطوة: صف بإيجاز تعابير الوجه أو لغة الجسد والعناصر السياقية التي تدل على الشعور )سطرين على الأكثر(.\n٢( استنتج الشعور الأساسي الظاهر باستخدام كلمة واحدة فقط من القائمة:\nسعادة، ثقة، خوف، مفاجأة، حزن، قرف، غضب، ترقب، محايد.\n٣( اطبع الإجابة النهائية في سطر منفصل بصيغة:\nالشعور: >الكلمة<\nابدأ الآن."
    return [
        {"role": "user", "parts": [
            {"text": system_prompt + "\n\n" + user_prompt},
            {"inline_data": {"mime_type": "image/png", "data": img_binary}}
        ]}
    ]

def query_gpt4o(image: Image.Image, prompt, prompt_type: str, max_retries=3, temperature=0.0, few_shot_examples=None) -> dict:
    for attempt in range(max_retries + 1):
        try:
            if prompt_type == 'zero_shot':
                messages = build_gpt4o_zero_shot_message(image)
            elif prompt_type == 'few_shot':
                messages = build_gpt4o_few_shot_message(image, few_shot_examples)
            elif prompt_type == 'chain_of_thought':
                messages = build_gpt4o_cot_message(image)
            else:
                raise ValueError(f"Unknown prompt_type: {prompt_type}")
                
            # Create a simplified version of the request JSON for logging
            # Remove base64 image data to prevent bloat
            request_json = []
            for msg in messages:
                if msg['role'] == 'user' and isinstance(msg['content'], list):
                    # For messages with image content, replace base64 with placeholder
                    simplified_content = []
                    for item in msg['content']:
                        if item['type'] == 'image_url':
                            simplified_content.append({'type': 'image_url', 'image_url': {'url': '[BASE64_IMAGE_DATA]'}})
                        else:
                            simplified_content.append(item)
                    simplified_msg = {'role': msg['role'], 'content': simplified_content}
                    request_json.append(simplified_msg)
                else:
                    request_json.append(msg)
            
            # Convert to JSON string for storage
            request_json_str = json.dumps(request_json, ensure_ascii=False, indent=2)
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=256,
                temperature=temperature + (attempt * 0.1)
            )
            answer = response.choices[0].message.content.strip()
            if is_refusal_message(answer) and attempt < max_retries:
                print(f"Refusal detected: '{answer}'. Retrying with same prompt (attempt {attempt+1}/{max_retries})...")
                continue
            if prompt_type == 'chain_of_thought':
                if 'الشعور:' in answer:
                    reasoning, label = answer.rsplit('الشعور:', 1)
                    return {'label': label.strip(), 'reasoning': reasoning.strip(), 'request_json': request_json_str}
                else:
                    return {'label': answer, 'reasoning': None, 'request_json': request_json_str}
            else:
                return {'label': answer, 'reasoning': None, 'request_json': request_json_str}
        except Exception as e:
            print(f"[ERROR] GPT-4o API call failed: {e}")
            if attempt < max_retries:
                print(f"Retrying due to error (attempt {attempt+1}/{max_retries})...")
                continue
            return {'label': None, 'reasoning': None, 'request_json': None}
    return {'label': "لم يتمكن النموذج من تحليل الصورة", 'reasoning': None, 'request_json': None}

def query_gemini(image: Image.Image, prompt, prompt_type: str, temperature=0.0, few_shot_examples=None) -> dict:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')
        if prompt_type == 'zero_shot':
            contents = build_gemini_zero_shot_content(image)
        elif prompt_type == 'few_shot':
            contents = build_gemini_few_shot_content(image, few_shot_examples)
        elif prompt_type == 'chain_of_thought':
            contents = build_gemini_cot_content(image)
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")
            
        # Create a simplified version of the request JSON for logging
        # Remove binary image data to prevent bloat
        request_json = []
        for msg in contents:
            if 'parts' in msg:
                simplified_parts = []
                for part in msg['parts']:
                    if 'inline_data' in part:
                        simplified_parts.append({'inline_data': {'mime_type': 'image/png', 'data': '[BINARY_IMAGE_DATA]'}})
                    else:
                        simplified_parts.append(part)
                simplified_msg = {'role': msg['role'], 'parts': simplified_parts}
                request_json.append(simplified_msg)
            else:
                request_json.append(msg)
                
        # Convert to JSON string for storage
        request_json_str = json.dumps(request_json, ensure_ascii=False, indent=2)
        
        response = model.generate_content(contents, generation_config={"temperature": temperature})
        answer = response.text.strip()
        if prompt_type == 'chain_of_thought':
            if 'الشعور:' in answer:
                reasoning, label = answer.rsplit('الشعور:', 1)
                return {'label': label.strip(), 'reasoning': reasoning.strip(), 'request_json': request_json_str}
            else:
                return {'label': answer, 'reasoning': None, 'request_json': request_json_str}
        else:
            return {'label': answer, 'reasoning': None, 'request_json': request_json_str}
    except Exception as e:
        print(f"[ERROR] Gemini API call failed: {e}")
        return {'label': None, 'reasoning': None, 'request_json': None} 