import os
from PIL import Image
import torch
import numpy as np
from shared_models import yolo_model, blip_processor, blip_model, clip_model, clip_preprocess

class MultimodalExtractor:
    """Multimodal feature extraction: BLIP, CLIP, YOLO."""
    def __init__(self):
        self.yolo_model = yolo_model
        self.blip_processor = blip_processor
        self.blip_model = blip_model
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def extract_multimodal_features_for_frames(self, keyframes_dir, video_name):
        yolo_class_names = self.yolo_model.names
        results = []
        for filename in sorted(os.listdir(keyframes_dir)):
            if filename.endswith('.jpg') and filename.startswith(f'{video_name}_representative_'):
                frame_path = os.path.join(keyframes_dir, filename)
                image = Image.open(frame_path).convert("RGB")
                # BLIP caption
                blip_inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    blip_out = self.blip_model.generate(**blip_inputs, max_new_tokens=30)
                    blip_caption = self.blip_processor.decode(blip_out[0], skip_special_tokens=True)
                # CLIP embedding
                clip_img = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    clip_emb = self.clip_model.encode_image(clip_img).cpu().numpy().tolist()[0]
                # YOLO detection
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
                results.append({
                    "frame": filename,
                    "frame_path": frame_path,
                    "blip_caption": blip_caption,
                    "clip_embedding": clip_emb,
                    "yolo_objects": yolo_objs
                })
        return results

    def summarize_multimodal_features(self, multimodal_features):
        yolo_counter = {}
        blip_captions = []
        for frame in multimodal_features:
            for obj in frame['yolo_objects']:
                cls_name = obj['cls']
                yolo_counter[cls_name] = yolo_counter.get(cls_name, 0) + 1
            blip_captions.append(frame['blip_caption'])
        yolo_summary = ', '.join([f'{k}: {v}' for k, v in yolo_counter.items()])
        blip_summary = ' | '.join(blip_captions[:3])
        return yolo_summary, blip_summary 