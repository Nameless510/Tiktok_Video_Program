from ultralytics import YOLOWorld, YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import clip as openai_clip
import torch
import os

# Global models for GPU acceleration
# YOLO model - Using YOLO-World for better product detection
def load_yolo_model():
    """Load YOLO model with fallback options"""
    try:
        # Try to load YOLO-World first
        if os.path.exists("yolov8s-world.pt"):
            print("Loading YOLO-World model: yolov8s-world.pt")
            return YOLOWorld("yolov8s-world.pt")
        elif os.path.exists("yolo-world-s.pt"):
            print("Loading YOLO-World model: yolo-world-s.pt")
            return YOLOWorld("yolo-world-s.pt")
        else:
            # Fallback to YOLOv8x if YOLO-World not available
            print("Loading YOLOv8x model as fallback")
            return YOLO("yolov8x.pt")
    except Exception as e:
        print(f"Error loading YOLO-World model: {e}")
        # Final fallback to YOLOv8x
        print("Falling back to YOLOv8x model")
        return YOLO("yolov8x.pt")

yolo_model = load_yolo_model()

# BLIP processor and model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

# CLIP model and preprocess
clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")

# TikTok product classes - Enhanced for YOLO-World product detection
TIKTOK_PRODUCT_CLASSES = {
    # Fashion & Beauty
    'clothing', 'shoes', 'bag', 'jewelry', 'watch', 'accessories', 'hat', 'scarf', 'glove', 'sunglasses',
    'ring', 'necklace', 'bracelet', 'earring', 'belt', 'wallet', 'perfume', 'makeup', 'cosmetics',
    'lipstick', 'foundation', 'blush', 'eyeshadow', 'mascara', 'eyeliner', 'nail', 'skincare',
    'shampoo', 'conditioner', 'body wash', 'toothbrush', 'toothpaste', 'soap', 'razor', 'deodorant',
    'lotion', 'cream', 'serum', 'mask', 'supplement', 'vitamin',
    
    # Electronics
    'phone', 'laptop', 'tablet', 'camera', 'tv', 'headphone', 'speaker', 'mouse', 'keyboard', 'monitor',
    'printer', 'router', 'charger', 'cable', 'adapter', 'power bank', 'memory card', 'usb', 'hard drive',
    'ssd', 'camera lens', 'tripod', 'microphone', 'drone', 'projector', 'screen', 'remote', 'controller',
    'joystick', 'console', 'glasses', 'case',
    
    # Home & Kitchen
    'appliance', 'kitchen', 'furniture', 'home', 'cleaning', 'lighting', 'fan', 'air conditioner',
    'refrigerator', 'microwave', 'oven', 'vacuum', 'hair dryer', 'towel', 'blanket', 'pillow',
    'mattress', 'curtain', 'rug', 'decoration', 'plant', 'flower', 'umbrella', 'mop', 'broom',
    'bucket', 'sponge', 'brush', 'detergent', 'disinfectant', 'sanitizer',
    
    # Food & Beverages
    'food', 'snack', 'drink', 'medicine', 'bandage', 'thermometer',
    
    # Toys & Entertainment
    'toys', 'game', 'board game', 'puzzle', 'toy car', 'doll', 'lego', 'block', 'action figure',
    'plush', 'book', 'stationery',
    
    # Sports & Outdoor
    'sports', 'outdoor', 'car', 'bicycle', 'tool',
    
    # Pet Products
    'pet', 'pet food', 'pet toy', 'pet bed', 'pet clothes', 'pet leash', 'pet bowl', 'pet litter',
    'pet shampoo', 'pet brush', 'pet carrier', 'pet cage', 'pet house', 'pet scratching',
    'pet training', 'pet medicine', 'pet supplement', 'pet collar', 'pet harness', 'pet tag',
    'pet feeder', 'pet waterer', 'pet mat', 'pet blanket', 'pet towel', 'pet toothbrush',
    'pet toothpaste', 'pet deodorant', 'pet cleaner', 'pet comb', 'pet nail', 'pet scissors',
    'pet clipper', 'pet dryer', 'pet perfume', 'pet treat', 'pet snack', 'pet chew', 'pet bone',
    'pet stick', 'pet rope', 'pet ball', 'pet frisbee', 'pet tunnel', 'pet tent', 'pet backpack',
    'pet stroller', 'pet car seat', 'pet seat belt', 'pet ramp', 'pet stairs', 'pet fence',
    'pet gate', 'pet playpen', 'pet pool', 'pet fountain', 'pet filter', 'pet pump', 'pet heater',
    'pet cooler', 'pet humidifier', 'pet dehumidifier', 'pet air purifier', 'pet camera',
    'pet monitor', 'pet tracker', 'pet gps', 'pet smart', 'pet automatic', 'pet interactive',
    'pet training', 'pet clicker', 'pet whistle', 'pet bell',
    
    # Baby Products
    'baby',
    
    # Audio & Video
    'audio', 'video', 'recording', 'streaming', 'broadcasting'
} 