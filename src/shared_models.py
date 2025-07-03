from ultralytics import YOLOWorld, YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration, BitsAndBytesConfig
import clip as openai_clip
import torch
import os
from modelscope import AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download

# Global models for GPU acceleration
# YOLO model - Using YOLO-World for better product detection
def load_yolo_model():
    """Load YOLO model with fallback options"""
    try:
        # Try to load YOLO-World first
        if os.path.exists("models/yolov8s-world.pt"):
            print("Loading YOLO-World model: models/yolov8s-world.pt")
            return YOLOWorld("models/yolov8s-world.pt")
        elif os.path.exists("yolo-world-s.pt"):
            print("Loading YOLO-World model: yolo-world-s.pt")
            return YOLOWorld("yolo-world-s.pt")
        else:
            # Fallback to YOLOv8x if YOLO-World not available
            print("Loading YOLOv8x model as fallback")
            return YOLO("models/yolov8x.pt")
    except Exception as e:
        print(f"Error loading YOLO-World model: {e}")
        # Final fallback to YOLOv8x
        print("Falling back to YOLOv8x model")
        return YOLO("models/yolov8x.pt")

yolo_model = load_yolo_model()

# BLIP processor and model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

# CLIP model and preprocess
clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
# Ensure CLIP model has device attribute
if not hasattr(clip_model, 'device'):
    clip_model.device = next(clip_model.parameters()).device

# Qwen-VL model for multimodal analysis
def load_qwen_model():
    """Load Qwen-VL model for multimodal analysis"""
    try:
        print("Loading Qwen-VL model...")
        
        # Check if model is already downloaded locally
        local_model_path = "./models/qwen/Qwen-VL-Chat"
        if os.path.exists(local_model_path):
            print(f"Using local model: {local_model_path}")
            model_dir = local_model_path
        else:
            print("Downloading Qwen-VL model... This may take a few minutes.")
            model_dir = snapshot_download('qwen/Qwen-VL-Chat', cache_dir='./models')
            print(f"Model downloaded to: {model_dir}")
        
        # Copy SimSun.ttf to model directory if it doesn't exist there
        fonts_dir = os.path.join(os.getcwd(), 'fonts')
        simsun_font_path = os.path.join(fonts_dir, 'SimSun.ttf')
        model_font_path = os.path.join(model_dir, 'SimSun.ttf')
        
        if os.path.exists(simsun_font_path) and not os.path.exists(model_font_path):
            import shutil
            shutil.copy2(simsun_font_path, model_font_path)
            print(f"Copied SimSun.ttf to model directory")
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, 
            trust_remote_code=True,
            padding_side='left'
        )
        print("Tokenizer loaded successfully")
        
        print("Loading model...")
        
        if torch.cuda.is_available():
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    device_map="auto",
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True
                ).eval()
                print("✓ Model loaded in 8bit quantization with cpu offload!")
            except Exception as e:
                print(f"[Fallback] 8bit quantization+offload failed: {e}\nTrying to load model on CPU only...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    device_map="cpu"
                ).eval()
                print("✓ Model loaded on CPU only (no quantization)")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu"
            ).eval()
        
        # Fix generation config
        if hasattr(model, 'generation_config'):
            model.generation_config.chat_format = 'chatml'
            # Add missing attributes
            if not hasattr(model.generation_config, 'max_window_size'):
                model.generation_config.max_window_size = 6144
            if not hasattr(model.generation_config, 'chat_format'):
                model.generation_config.chat_format = 'chatml'
        
        print("✓ Qwen-VL model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Warning: Qwen-VL model failed to load: {e}")
        print("Trying alternative loading method...")
        
        try:
            # Alternative method with different settings
            local_model_path = "./models/qwen/Qwen-VL-Chat"
            if os.path.exists(local_model_path):
                model_dir = local_model_path
            else:
                model_dir = snapshot_download('qwen/Qwen-VL-Chat', cache_dir='./models')
            
            # Copy SimSun.ttf to model directory if it doesn't exist there
            fonts_dir = os.path.join(os.getcwd(), 'fonts')
            simsun_font_path = os.path.join(fonts_dir, 'SimSun.ttf')
            model_font_path = os.path.join(model_dir, 'SimSun.ttf')
            
            if os.path.exists(simsun_font_path) and not os.path.exists(model_font_path):
                import shutil
                shutil.copy2(simsun_font_path, model_font_path)
                print(f"Copied SimSun.ttf to model directory")
            
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cuda:0",  # 强制加载到 GPU
                fp16=True
            ).eval()
            
            # Fix generation config
            if hasattr(model, 'generation_config'):
                model.generation_config.chat_format = 'chatml'
                # Add missing attributes
                if not hasattr(model.generation_config, 'max_window_size'):
                    model.generation_config.max_window_size = 6144
                if not hasattr(model.generation_config, 'chat_format'):
                    model.generation_config.chat_format = 'chatml'
            
            print("✓ Qwen-VL model loaded on GPU successfully!")
            return model, tokenizer
            
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            print("The system will work without Qwen-VL functionality.")
            print("To enable Qwen-VL, ensure you have sufficient disk space and memory.")
            return None, None

qwen_model, qwen_tokenizer = load_qwen_model()
print("[DEBUG] Qwen model type:", type(qwen_model))
print("[DEBUG] Has chat method:", hasattr(qwen_model, 'chat'))
print("[DEBUG] Qwen tokenizer type:", type(qwen_tokenizer))

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

# TikTok product categorization hierarchy
TIKTOK_CATEGORIES = {
    "Beauty and Personal Care": {
        "Makeup": ["Foundation", "Concealer", "Powder", "Blush", "Bronzer", "Highlighter", "Eyeshadow", "Eyeliner", "Mascara", "Lipstick", "Lip gloss", "Lip liner", "Nail polish", "Makeup brushes", "Makeup sponges", "Makeup remover"],
        "Skincare": ["Cleanser", "Toner", "Serum", "Moisturizer", "Sunscreen", "Face mask", "Eye cream", "Acne treatment", "Anti-aging", "Exfoliator", "Face oil", "Face mist"],
        "Hair Care": ["Shampoo", "Conditioner", "Hair mask", "Hair oil", "Hair spray", "Hair gel", "Hair wax", "Hair dye", "Hair extensions", "Hair accessories"],
        "Fragrance": ["Perfume", "Body spray", "Deodorant", "Body lotion", "Body wash", "Soap"],
        "Tools": ["Hair dryer", "Straightener", "Curling iron", "Makeup mirror", "Tweezers", "Nail clippers", "Razor", "Electric shaver"]
    },
    "Fashion": {
        "Clothing": ["Dresses", "Tops", "Bottoms", "Outerwear", "Activewear", "Lingerie", "Swimwear", "Shoes", "Bags", "Accessories"],
        "Jewelry": ["Necklaces", "Earrings", "Bracelets", "Rings", "Watches", "Anklets"],
        "Accessories": ["Hats", "Scarves", "Belts", "Sunglasses", "Gloves", "Wallets", "Phone cases"]
    },
    "Electronics": {
        "Mobile Devices": ["Smartphones", "Tablets", "Laptops", "Smartwatches", "Earbuds", "Headphones"],
        "Home Electronics": ["TVs", "Speakers", "Cameras", "Gaming consoles", "Smart home devices"],
        "Accessories": ["Chargers", "Cables", "Cases", "Screen protectors", "Power banks"]
    },
    "Home and Garden": {
        "Kitchen": ["Appliances", "Cookware", "Utensils", "Storage", "Decor"],
        "Furniture": ["Living room", "Bedroom", "Office", "Outdoor", "Storage"],
        "Decor": ["Lighting", "Art", "Plants", "Rugs", "Curtains", "Candles"]
    },
    "Health and Wellness": {
        "Fitness": ["Exercise equipment", "Workout clothes", "Supplements", "Fitness trackers"],
        "Nutrition": ["Vitamins", "Protein powder", "Superfoods", "Healthy snacks"],
        "Wellness": ["Essential oils", "Meditation apps", "Yoga mats", "Massage tools"]
    },
    "Food and Beverages": {
        "Snacks": ["Chips", "Nuts", "Candy", "Chocolate", "Dried fruits"],
        "Beverages": ["Coffee", "Tea", "Juice", "Soda", "Energy drinks", "Water"],
        "Cooking": ["Ingredients", "Spices", "Oils", "Sauces", "Condiments"]
    },
    "Toys and Entertainment": {
        "Toys": ["Educational toys", "Building blocks", "Dolls", "Action figures", "Board games"],
        "Books": ["Fiction", "Non-fiction", "Children's books", "Educational"],
        "Hobbies": ["Arts and crafts", "Collectibles", "Musical instruments"]
    },
    "Sports and Outdoor": {
        "Sports Equipment": ["Balls", "Rackets", "Weights", "Yoga mats", "Running gear"],
        "Outdoor": ["Camping gear", "Hiking equipment", "Bicycles", "Skateboards"],
        "Fitness": ["Gym equipment", "Workout clothes", "Sports shoes"]
    },
    "Baby and Kids": {
        "Baby Care": ["Diapers", "Baby food", "Baby clothes", "Baby toys", "Baby gear"],
        "Kids Fashion": ["Children's clothing", "Kids shoes", "Kids accessories"],
        "Education": ["Learning toys", "Books", "School supplies"]
    },
    "Pet Supplies": {
        "Pet Food": ["Dog food", "Cat food", "Bird food", "Fish food"],
        "Pet Care": ["Grooming supplies", "Toys", "Beds", "Carriers"],
        "Pet Health": ["Vitamins", "Medicine", "Dental care", "Flea treatment"]
    }
} 