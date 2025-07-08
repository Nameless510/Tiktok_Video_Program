from ultralytics import YOLOWorld, YOLO
import torch
import os
import sys
import tempfile
import yaml
import dora


# Import transformers with error handling
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration, BitsAndBytesConfig
    print("Transformers imported successfully")
except ImportError as e:
    print(f"Error importing transformers: {e}")
    BlipProcessor = None
    BlipForConditionalGeneration = None
    BitsAndBytesConfig = None

# Import CLIP
try:
    import clip as openai_clip
    print("CLIP imported successfully")
except ImportError as e:
    print(f"Error importing CLIP: {e}")
    openai_clip = None

# Import modelscope
try:
    from modelscope import AutoTokenizer, AutoModelForCausalLM
    from modelscope import snapshot_download
    print("ModelScope imported successfully")
except ImportError as e:
    print(f"Error importing ModelScope: {e}")
    AutoTokenizer = None
    AutoModelForCausalLM = None
    snapshot_download = None

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
if BlipProcessor is not None and BlipForConditionalGeneration is not None:
    try:
        print("Loading BLIP model...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = blip_model.to("cuda" if torch.cuda.is_available() else "cpu")
        print("BLIP model loaded successfully")
    except Exception as e:
        print(f"Error loading BLIP model: {e}")
        print("Setting BLIP model to None - some features may be limited")
        blip_processor = None
        blip_model = None
else:
    print("BLIP modules not available, setting to None")
    blip_processor = None
    blip_model = None

# CLIP model and preprocess
if openai_clip is not None:
    try:
        print("Loading CLIP model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device=device)
        # Ensure CLIP model has device attribute
        if not hasattr(clip_model, 'device'):
            clip_model.device = device
        print("CLIP model loaded successfully")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        print("Setting CLIP model to None - some features may be limited")
        clip_model = None
        clip_preprocess = None
else:
    print("CLIP module not available, setting to None")
    clip_model = None
    clip_preprocess = None

# Qwen model loading section
qwen_model = None
qwen_tokenizer = None
# try:
#     from transformers import AutoTokenizer as HF_AutoTokenizer, AutoModelForCausalLM as HF_AutoModelForCausalLM
#     print("Transformers imported for Qwen.")
#     model_dir = os.path.abspath("models/qwen/Qwen-VL-Chat")
#     print(f"Loading Qwen model from local path (transformers): {model_dir}")
#     qwen_tokenizer = HF_AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
#     if qwen_tokenizer.pad_token is None:
#         qwen_tokenizer.pad_token = "<|extra_0|>"
#         qwen_tokenizer.pad_token_id = qwen_tokenizer.convert_tokens_to_ids("<|extra_0|>")
#     elif qwen_tokenizer.pad_token == qwen_tokenizer.eos_token:
#         print("Warning: pad_token equals eos_token, setting pad_token to <|extra_0|>")
#         qwen_tokenizer.pad_token = "<|extra_0|>"
#         qwen_tokenizer.pad_token_id = qwen_tokenizer.convert_tokens_to_ids("<|extra_0|>")
#     if torch.cuda.is_available():
#         print("Using CUDA for Qwen model (transformers)")
#         qwen_model = HF_AutoModelForCausalLM.from_pretrained(
#             model_dir, device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.float16
#         ).eval()
#     else:
#         print("CUDA not available, using CPU for Qwen model (transformers)")
#         qwen_model = HF_AutoModelForCausalLM.from_pretrained(
#             model_dir, device_map="cpu", trust_remote_code=True, torch_dtype=torch.float32
#         ).eval()
#     if hasattr(qwen_model, 'generation_config'):
#         qwen_model.generation_config.chat_format = 'chatml'
#         qwen_model.generation_config.max_window_size = 6144
#     print("Qwen model loaded successfully (transformers)")
# except Exception as e:
#     print(f"Transformers Qwen load failed: {e}")
#     import traceback; traceback.print_exc()
#     qwen_model = None
#     qwen_tokenizer = None

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

# === Audio Models (Whisper, Silero-VAD, Demucs, Shazam) ===

os.environ['NUMBA_CACHE_DIR'] = tempfile.gettempdir()

def pip_install(package):
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"Failed to install {package}: {e}")

whisper_model = None
try:
    import whisper
    # print("Loading Whisper model (large-v3)...")
    # try:
    #     whisper_model = whisper.load_model("large-v3", device="cuda" if torch.cuda.is_available() else "cpu")
    #     print("Whisper large-v3 loaded on CUDA")
    # except RuntimeError as e:
    #     print(f"Whisper large-v3 CUDA failed: {e}")
    #     print("Trying large-v3 on CPU...")
    #     try:
    #         whisper_model = whisper.load_model("large-v3", device="cpu")
    #         print("Whisper large-v3 loaded on CPU")
    #     except Exception as e:
    #         print(f"Whisper large-v3 CPU failed: {e}")
    #         print("Trying medium on CPU...")
    #         try:
    #             whisper_model = whisper.load_model("medium", device="cpu")
    #             print("Whisper medium loaded on CPU")
    #         except Exception as e:
    #             print(f"Whisper medium failed: {e}")
    #             print("Trying base on CPU...")
    #             try:
    #                 whisper_model = whisper.load_model("base", device="cpu")
    #                 print("Whisper base loaded on CPU")
    #             except Exception as e:
    #                 print(f"Whisper base failed: {e}")
    #                 whisper_model = None
    print("Loading Whisper model (small)...")
    whisper_model = whisper.load_model("small", device="cuda" if torch.cuda.is_available() else "cpu")
    print("Whisper small loaded!")
except ImportError as e:
    print(f"Error importing whisper: {e}")
    whisper_model = None

# ========== Silero VAD ==========
silero_vad_model = None
try:
    vad_package = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    silero_vad_model = vad_package[0] if isinstance(vad_package, tuple) else vad_package
    silero_vad_model = silero_vad_model.to("cuda" if torch.cuda.is_available() else "cpu")
    silero_vad_model.eval()
    print("Silero VAD model loaded successfully")
except Exception as e:
    print(f"Error loading Silero VAD model: {e}")
    silero_vad_model = None

# ========== Demucs (Hybrid) ==========
demucs_model = None
try:
    try:
        from demucs.pretrained import get_model
    except ImportError:
        get_model = None
    if get_model is not None:
        print("Loading Demucs model (pip version)...")
        demucs_model = get_model("htdemucs")
        if hasattr(demucs_model, 'to'):
            demucs_model = demucs_model.to("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(demucs_model, 'eval'):
            demucs_model.eval()
        print("Demucs model loaded successfully")
    else:
        print("demucs.pretrained.get_model not available.")
        demucs_model = None
except Exception as e:
    print(f"Error loading Demucs model: {e}")
    print("Demucs will be unavailable.")
    demucs_model = None

# ========== Shazamio (for music recognition) ==========
shazamio = None
shazam_client = None
try:
    try:
        import shazamio
        from shazamio import Shazam
    except ImportError:
        shazamio = None
        Shazam = None
    if shazamio is not None and Shazam is not None:
        print("Shazamio imported successfully")
        shazam_client = Shazam()
    else:
        print("shazamio not found, installing...")
        pip_install('shazamio')
        try:
            import shazamio
            from shazamio import Shazam
            print("Shazamio installed and imported successfully")
            shazam_client = Shazam()
        except Exception as e:
            print(f"Shazamio still not available: {e}")
            shazamio = None
            shazam_client = None
except Exception as e:
    print(f"Error loading Shazamio: {e}")
    shazamio = None
    shazam_client = None

__all__ = [
    'yolo_model', 'blip_processor', 'blip_model', 'clip_model', 'clip_preprocess',
    'qwen_model', 'qwen_tokenizer',
    'whisper_model', 'silero_vad_model', 'demucs_model', 'shazam_client'
] 