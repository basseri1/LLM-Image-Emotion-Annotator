import os
from PIL import Image
import base64
import io

def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith((
            '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp', '.ico', '.heic'  # Extend as needed
        )):
            path = os.path.join(directory, filename)
            try:
                img = Image.open(path)
                images.append((path, img))
            except Exception as e:
                print(f'Error loading {path}: {e}')
    return images

def encode_image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """
    Encode a PIL Image to a base64 string.
    Args:
        image: PIL Image object
        format: Format to encode (default: 'PNG')
    Returns:
        Base64-encoded string of the image
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_b64 