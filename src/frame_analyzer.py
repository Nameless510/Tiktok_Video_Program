import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
from skimage.metrics import structural_similarity as ssim
from shared_models import yolo_model, TIKTOK_PRODUCT_CLASSES

class FrameAnalyzer:
    """Frame analysis: filtering, clustering, and product detection."""
    def __init__(self):
        self.yolo_model = yolo_model

    def filter_similar_keyframes(self, keyframes_dir, video_name, ssim_threshold=0.95):
        if not os.path.exists(keyframes_dir):
            return []
        frame_files = [f for f in sorted(os.listdir(keyframes_dir)) if f.endswith('.jpg') and f.startswith(f'{video_name}_keyframe_')]
        if len(frame_files) <= 1:
            return [os.path.join(keyframes_dir, f) for f in frame_files]
        filtered_frames = []
        previous_frame_path = None
        for frame_file in frame_files:
            current_frame_path = os.path.join(keyframes_dir, frame_file)
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
                    print(f"  - Removed similar frame: {frame_file} (SSIM: {ssim_score:.3f})")
            except Exception as e:
                print(f"Error comparing frames {frame_file}: {e}")
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

    def get_representative_frames(self, keyframes_dir, video_name):
        """
        Select 2-5 representative frames using YOLO detection first, then KMeans clustering.
        1. Filter out black/blank frames, then blurry frames.
        2. Extract YOLO features for all valid frames.
        3. Separate product frames and non-product frames.
        4. Cluster product frames first, then non-product frames if needed.
        5. Select representative frames prioritizing product frames.
        """
        if not os.path.exists(keyframes_dir):
            return []
        
        # Step 1: Get all frame files and filter basic issues
        frame_files = [f for f in sorted(os.listdir(keyframes_dir)) 
                      if f.endswith('.jpg') and f.startswith(f'{video_name}_keyframe_')]
        
        if not frame_files:
            return []
        
        # Step 2: Filter black and blurry frames
        valid_frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(keyframes_dir, frame_file)
            if not self.is_black_frame(frame_path) and not self.is_blurry(frame_path):
                valid_frames.append(frame_path)
        
        if not valid_frames:
            return []
        
        # Step 3: Extract YOLO features for all valid frames
        frame_infos = []
        for frame_path in valid_frames:
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
        
        # Step 4: Separate product frames and non-product frames
        product_frames = [info for info in frame_infos if self.is_tiktok_product_frame(info['yolo_objects'])]
        non_product_frames = [info for info in frame_infos if not self.is_tiktok_product_frame(info['yolo_objects'])]
        
        print(f"  - Found {len(product_frames)} product frames, {len(non_product_frames)} non-product frames")
        
        # Step 5: Determine number of clusters needed
        n_clusters = self.get_representative_frame_count(len(frame_infos))
        
        # Step 6: Select representative frames
        representative_frames = []
        
        # First, try to select from product frames
        if product_frames:
            if len(product_frames) <= n_clusters:
                # Use all product frames
                representative_frames = [f['frame_path'] for f in product_frames]
            else:
                # Cluster product frames to select best ones
                product_features = self._extract_color_features([f['frame_path'] for f in product_frames])
                if len(product_features) > 0:
                    kmeans = MiniBatchKMeans(n_clusters=min(n_clusters, len(product_frames)), 
                                           random_state=42, batch_size=100)
                    cluster_labels = kmeans.fit_predict(product_features)
                    
                    for i in range(min(n_clusters, len(product_frames))):
                        cluster_indices = [j for j in range(len(product_frames)) if cluster_labels[j] == i]
                        cluster_frames = product_features[cluster_indices]
                        distances = euclidean_distances(cluster_frames, [kmeans.cluster_centers_[i]])
                        closest_idx = np.argmin(distances)
                        representative_frames.append(product_frames[cluster_indices[closest_idx]]['frame_path'])
        
        # If we need more frames, add from non-product frames
        if len(representative_frames) < n_clusters and non_product_frames:
            remaining_needed = n_clusters - len(representative_frames)
            if len(non_product_frames) <= remaining_needed:
                representative_frames.extend([f['frame_path'] for f in non_product_frames])
            else:
                # Cluster non-product frames to select best ones
                non_product_features = self._extract_color_features([f['frame_path'] for f in non_product_frames])
                if len(non_product_features) > 0:
                    kmeans = MiniBatchKMeans(n_clusters=remaining_needed, random_state=42, batch_size=100)
                    cluster_labels = kmeans.fit_predict(non_product_features)
                    
                    for i in range(remaining_needed):
                        cluster_indices = [j for j in range(len(non_product_frames)) if cluster_labels[j] == i]
                        cluster_frames = non_product_features[cluster_indices]
                        distances = euclidean_distances(cluster_frames, [kmeans.cluster_centers_[i]])
                        closest_idx = np.argmin(distances)
                        representative_frames.append(non_product_frames[cluster_indices[closest_idx]]['frame_path'])
        
        # If still not enough frames, pad by repeating the last one
        while len(representative_frames) < n_clusters:
            if representative_frames:
                representative_frames.append(representative_frames[-1])
            else:
                # Fallback to first valid frame
                representative_frames.append(valid_frames[0])
        
        return representative_frames[:n_clusters]
    
    def _extract_color_features(self, frame_paths):
        """Extract color histogram features for clustering."""
        features = []
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist = hist.flatten() / hist.sum()
                features.append(hist)
        return np.array(features) if features else np.array([])

    def save_representative_frames(self, representative_frames, output_dir, video_name):
        os.makedirs(output_dir, exist_ok=True)
        for old_file in os.listdir(output_dir):
            if old_file.startswith(f'{video_name}_representative_'):
                os.remove(os.path.join(output_dir, old_file))
        rep_set = set(os.path.abspath(f) for f in representative_frames)
        idx_map = {os.path.abspath(f): i for i, f in enumerate(representative_frames)}
        for file in os.listdir(output_dir):
            if file.endswith('.jpg') and file.startswith(f'{video_name}_keyframe_'):
                file_path = os.path.join(output_dir, file)
                abs_path = os.path.abspath(file_path)
                if abs_path in rep_set:
                    idx = idx_map[abs_path]
                    new_filename = f'{video_name}_representative_{idx+1:02d}.jpg'
                    new_path = os.path.join(output_dir, new_filename)
                    os.rename(file_path, new_path)
                    print(f"  - Saved representative frame: {new_filename}")
                else:
                    os.remove(file_path) 