import os
from PIL import Image
import base64
import io

def resize_image_preserve_aspect_ratio(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Resize an image so that the longer edge is limited to max_size pixels, 
    preserving the aspect ratio.
    
    Args:
        image: PIL Image object
        max_size: Maximum size for the longer edge (default: 1024)
        
    Returns:
        Resized PIL Image with aspect ratio preserved
    """
    # Get the original dimensions
    width, height = image.size
    
    # If the image is already smaller than max_size in both dimensions, return it as is
    if width <= max_size and height <= max_size:
        return image
    
    # Determine which dimension is longer and calculate scaling factor
    if width > height:
        # Width is the longer edge
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        # Height is the longer edge
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Resize the image with the calculated dimensions
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image

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
    Resizes the image to a maximum of 1024 pixels on the longer edge while preserving aspect ratio.
    
    Args:
        image: PIL Image object
        format: Format to encode (default: 'PNG')
        
    Returns:
        Base64-encoded string of the resized image
    """
    # Resize image preserving aspect ratio
    resized_image = resize_image_preserve_aspect_ratio(image, max_size=1024)
    
    # Encode the resized image
    buffered = io.BytesIO()
    resized_image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_b64 