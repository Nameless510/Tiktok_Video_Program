import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from shared_models import yolo_model, clip_model, clip_preprocess, TIKTOK_PRODUCT_CLASSES
import torch
from PIL import Image
import shutil
import json
import re

class FrameAnalyzer:
    """Frame analysis: filtering, clustering, and product detection."""
    def __init__(self):
        self.yolo_model = yolo_model

    def filter_similar_keyframes(self, frame_dir, video_name, ssim_threshold=0.8):
        """
        Filter out visually similar frames using SSIM. Accepts a directory of frames.
        """
        frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.jpg') or f.endswith('.png')])
        if not frame_files or len(frame_files) <= 1:
            return frame_files
        filtered_frames = []
        previous_frame_path = None
        for current_frame_path in frame_files:
            if not os.path.exists(current_frame_path):
                continue
            if previous_frame_path is None:
                filtered_frames.append(current_frame_path)
                previous_frame_path = current_frame_path
                continue
            try:
                current_frame = cv2.imread(current_frame_path)
                previous_frame = cv2.imread(previous_frame_path)
                if current_frame is None or previous_frame is None:
                    continue
                height, width = current_frame.shape[:2]
                previous_frame_resized = cv2.resize(previous_frame, (width, height))
                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                previous_gray = cv2.cvtColor(previous_frame_resized, cv2.COLOR_BGR2GRAY)
                ssim_score = ssim(current_gray, previous_gray)
                if ssim_score < ssim_threshold:
                    filtered_frames.append(current_frame_path)
                    previous_frame_path = current_frame_path
                else:
                    os.remove(current_frame_path)
                    print(f"  - Removed similar frame: {os.path.basename(current_frame_path)} (SSIM: {ssim_score:.3f})")
            except Exception as e:
                print(f"Error comparing frames {current_frame_path}: {e}")
                filtered_frames.append(current_frame_path)
                previous_frame_path = current_frame_path
        return filtered_frames

    def is_black_frame(self, image_path, brightness_threshold=30):
        img = cv2.imread(image_path)
        if img is None:
            return True
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        return mean_brightness < brightness_threshold

    def is_blurry(self, image_path, threshold=50):
        """Return True if the image is blurry (Laplacian variance below threshold)."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
        lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
        return lap_var < threshold

    def filter_blurry_frames(self, frame_infos, threshold=50, keep_n=5):
        """Filter out blurry frames. If all are blurry, keep the top-n clearest frames."""
        filtered = []
        for info in frame_infos:
            if not self.is_blurry(info['frame_path'], threshold=threshold):
                filtered.append(info)
        if not filtered and frame_infos:
            # If all are blurry, keep the top-n clearest
            scored = []
            for info in frame_infos:
                img = cv2.imread(info['frame_path'], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
                else:
                    lap_var = 0
                scored.append((lap_var, info))
            scored.sort(reverse=True, key=lambda x: x[0])
            filtered = [x[1] for x in scored[:min(keep_n, len(scored))]]
        return filtered

    def extract_yolo_features_for_keyframes(self, keyframes_dir, video_name):
        yolo_class_names = self.yolo_model.names
        frame_infos = []
        for filename in sorted(os.listdir(keyframes_dir)):
            if filename.endswith('.jpg') and filename.startswith(f'{video_name}_keyframe_'):
                frame_path = os.path.join(keyframes_dir, filename)
                if not os.path.exists(frame_path):
                    continue
                yolo_results = self.yolo_model(frame_path)
                yolo_objs = []
                for r in yolo_results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = yolo_class_names.get(cls_id, str(cls_id))
                        yolo_objs.append({
                            "cls": cls_name,
                            "conf": float(box.conf[0]),
                            "xyxy": [float(x) for x in box.xyxy[0]]
                        })
                frame_infos.append({
                    "frame": filename,
                    "frame_path": frame_path,
                    "yolo_objects": yolo_objs
                })
        return frame_infos

    def is_tiktok_product_frame(self, yolo_objects):
        return any(obj['cls'].lower() in TIKTOK_PRODUCT_CLASSES for obj in yolo_objects)

    def filter_contentless_frames(self, frame_infos):
        filtered = []
        for info in frame_infos:
            if self.is_black_frame(info['frame_path']):
                continue
            filtered.append(info)
        return filtered

    def get_representative_frame_count(self, num_keyframes):
        # Always return 2-5 based on total valid frames
        if num_keyframes <= 2:
            return 2
        elif num_keyframes <= 5:
            return num_keyframes
        elif num_keyframes <= 15:
            return 3
        elif num_keyframes <= 30:
            return 4
        else:
            return 5

    def get_representative_frames(self, frame_dir, video_name):
        """
        Select 2-5 representative frames using YOLO detection and CLIP similarity.
        Accepts a directory of frames.
        """
        frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.jpg') or f.endswith('.png')])
        if not frame_files:
            return []
        # Step 1: Filter black and blurry frames
        valid_frames = []
        for frame_path in frame_files:
            if not os.path.exists(frame_path):
                continue
            if not self.is_black_frame(frame_path) and not self.is_blurry(frame_path):
                valid_frames.append(frame_path)
        if not valid_frames:
            return []
        # Step 2: Extract YOLO features for all valid frames
        frame_infos = []
        for frame_path in valid_frames:
            if not os.path.exists(frame_path):
                continue
            yolo_results = self.yolo_model(frame_path)
            yolo_objs = []
            for r in yolo_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.yolo_model.names.get(cls_id, str(cls_id))
                    yolo_objs.append({
                        "cls": cls_name,
                        "conf": float(box.conf[0]),
                        "xyxy": [float(x) for x in box.xyxy[0]]
                    })
            frame_infos.append({
                "frame_path": frame_path,
                "yolo_objects": yolo_objs
            })
        # Step 3: Separate product frames and non-product frames
        product_frames = [info for info in frame_infos if self.is_tiktok_product_frame(info['yolo_objects'])]
        non_product_frames = [info for info in frame_infos if not self.is_tiktok_product_frame(info['yolo_objects'])]
        print(f"  - Found {len(product_frames)} product frames, {len(non_product_frames)} non-product frames")
        # Step 4: Determine number of frames needed
        n_frames = self.get_representative_frame_count(len(frame_infos))
        # Step 5: Select representative frames using CLIP
        representative_frames = []
        product_prompts = [
            "product", "item", "goods", "merchandise", "commodity",
            "clothing", "electronics", "accessories", "beauty products",
            "fashion", "shopping", "retail", "commercial"
        ]
        if product_frames:
            if len(product_frames) <= n_frames:
                representative_frames = [f['frame_path'] for f in product_frames]
            else:
                selected_product_frames = self._select_frames_with_clip(
                    [f['frame_path'] for f in product_frames], 
                    product_prompts, 
                    min(n_frames, len(product_frames))
                )
                representative_frames = selected_product_frames
        if len(representative_frames) < n_frames and non_product_frames:
            remaining_needed = n_frames - len(representative_frames)
            if len(non_product_frames) <= remaining_needed:
                representative_frames.extend([f['frame_path'] for f in non_product_frames])
            else:
                selected_non_product_frames = self._select_frames_with_clip(
                    [f['frame_path'] for f in non_product_frames], 
                    product_prompts, 
                    remaining_needed
                )
                representative_frames.extend(selected_non_product_frames)
        while len(representative_frames) < n_frames:
            if representative_frames:
                representative_frames.append(representative_frames[-1])
            else:
                representative_frames.append(valid_frames[0])
        return representative_frames[:n_frames]
    
    def _select_frames_with_clip(self, frame_paths, text_prompts, n_select):
        """
        Use CLIP to select frames most similar to product-related text prompts.
        """
        if not frame_paths or n_select <= 0:
            return []
        
        try:
            # Get CLIP model device safely
            from shared_models import clip_model, clip_preprocess
            clip_device = next(clip_model.parameters()).device
            
            # Encode text prompts - use tokenize for text
            import clip as openai_clip
            text_tokens = openai_clip.tokenize(text_prompts).to(clip_device)
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Encode images
            image_features = []
            valid_frame_paths = []
            
            for frame_path in frame_paths:
                if not os.path.exists(frame_path):
                    continue
                try:
                    # Load and preprocess image for CLIP
                    image = Image.open(frame_path).convert('RGB')
                    image_input = clip_preprocess(image).unsqueeze(0).to(clip_device)
                    image_feature = clip_model.encode_image(image_input)
                    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
                    image_features.append(image_feature)
                    valid_frame_paths.append(frame_path)
                except Exception as e:
                    print(f"Error processing image {frame_path}: {e}")
                    continue
            
            if not image_features:
                return frame_paths[:n_select]
            
            # Calculate similarities
            image_features = torch.cat(image_features, dim=0)
            similarities = torch.matmul(image_features, text_features.T)
            
            # Get average similarity across all prompts
            avg_similarities = similarities.mean(dim=1)
            
            # Select top n frames
            top_indices = torch.argsort(avg_similarities, descending=True)[:n_select]
            selected_frames = [valid_frame_paths[i] for i in top_indices]
            
            return selected_frames
            
        except Exception as e:
            print(f"Error in CLIP selection: {e}")
            # Fallback to simple selection
            return frame_paths[:n_select]

    def save_representative_frames(self, representative_frames, video_dir, video_name):
        import re
        try:
            rep_filenames = set()
            pattern = re.compile(r'_(\d+\.\d+)\.jpg$', re.IGNORECASE)
            for frame_path in representative_frames:
                fname = os.path.basename(frame_path)
                match = pattern.search(fname)
                if match:
                    timestamp = match.group(1)
                    new_filename = f"representative_{timestamp}.jpg"
                else:
                    new_filename = f"representative_{fname}"
                new_path = os.path.join(video_dir, new_filename)
                if os.path.abspath(frame_path) != os.path.abspath(new_path):
                    shutil.move(frame_path, new_path)
                rep_filenames.add(new_filename.lower())

            for fname in os.listdir(video_dir):
                ext = fname.lower().split('.')[-1]
                fpath = os.path.join(video_dir, fname)
                if ext in ('jpg', 'jpeg', 'png') and fname.lower() not in rep_filenames:
                    try:
                        os.remove(fpath)
                    except Exception as e:
                        print(f"  - Warning: Could not remove {fpath}: {e}")
            print('After cleanup, video_dir files:', os.listdir(video_dir))
        except Exception as e:
            print(f"Error saving representative frames: {e}")

    def cleanup_frames(self, output_dir, keep_files):
        keep_set = set(os.path.abspath(f) for f in keep_files)
        for fname in os.listdir(output_dir):
            if fname.endswith('.jpg') or fname.endswith('.png'):
                fpath = os.path.abspath(os.path.join(output_dir, fname))
                if fpath not in keep_set:
                    try:
                        os.remove(fpath)
                    except Exception as e:
                        print(f"  - Warning: Could not remove {fname}: {e}") 

    def save_representative_timestamps(self, video_dir, video_name):
        import re
        rep_info = []
        pattern = re.compile(r'representative_(\d+\.\d+)\.jpg$', re.IGNORECASE)
        for fname in os.listdir(video_dir):
            match = pattern.match(fname)
            if match:
                timestamp = float(match.group(1))
                rep_info.append({'file': fname, 'timestamp': timestamp})
        out_path = os.path.join(video_dir, 'representative_timestamps.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(rep_info, f, indent=2, ensure_ascii=False) 