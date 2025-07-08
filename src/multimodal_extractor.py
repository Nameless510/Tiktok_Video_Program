import os
from PIL import Image
import torch
import numpy as np
import pandas as pd
import json
import re
from shared_models import (
    yolo_model, blip_processor, blip_model, clip_model, clip_preprocess,
    qwen_model, qwen_tokenizer, TIKTOK_CATEGORIES
)
import time

class MultimodalExtractor:
    """Enhanced multimodal feature extraction using all available models effectively."""
    def __init__(self):
        self.yolo_model = yolo_model
        self.blip_processor = blip_processor
        self.blip_model = blip_model
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.qwen_model = qwen_model
        self.qwen_tokenizer = qwen_tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tiktok_categories = TIKTOK_CATEGORIES

    def extract_comprehensive_features(self, image_path, ocr_text="", audio_text=""):
        """
        Extract comprehensive features using all available models.
        Returns detailed analysis with proper categorization.
        """
        try:
            # Step 1: Extract features from all models
            if not os.path.exists(image_path): print(f"Warning: file does not exist: {image_path}"); return
            yolo_objects = self._extract_yolo_features(image_path)
            blip_caption = self._extract_blip_caption(image_path)
            clip_embedding = self._extract_clip_embedding(image_path)
            
            # Step 2: Use Qwen for intelligent analysis with all context
            analysis_result = self._analyze_with_qwen_enhanced(
                image_path, yolo_objects, blip_caption, ocr_text, audio_text
            )
            
            return analysis_result
            
        except Exception as e:
            print(f"Error in comprehensive feature extraction: {e}")
            return self._create_fallback_result(image_path, ocr_text, audio_text)

    def _extract_yolo_features(self, image_path):
        """Extract YOLO object detection results."""
        try:
            if not os.path.exists(image_path): print(f"Warning: file does not exist: {image_path}"); return
            results = self.yolo_model(image_path)
            objects = []
            
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.yolo_model.names.get(cls_id, str(cls_id))
                    confidence = float(box.conf[0])
                    
                    if confidence > 0.3:  # Filter low confidence detections
                        objects.append({
                            "class": cls_name,
                            "confidence": confidence,
                            "bbox": [float(x) for x in box.xyxy[0]]
                        })
            
            return objects
        except Exception as e:
            print(f"Error in YOLO extraction: {e}")
            return []

    def _extract_blip_caption(self, image_path):
        """Extract BLIP image caption."""
        try:
            if not os.path.exists(image_path): print(f"Warning: file does not exist: {image_path}"); return
            image = Image.open(image_path).convert("RGB")
            inputs = self.blip_processor(image, return_tensors="pt")
            
            # Move inputs to device and ensure proper types
            device_inputs = {}
            for k, v in inputs.items():
                if hasattr(v, 'to'):
                    device_inputs[k] = v.to(self.device)
                else:
                    device_inputs[k] = v
            
            with torch.no_grad():
                outputs = self.blip_model.generate(**device_inputs, max_new_tokens=50)
                caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            
            return caption
        except Exception as e:
            print(f"Error in BLIP extraction: {e}")
            return ""

    def _extract_clip_embedding(self, image_path):
        """Extract CLIP image embedding."""
        try:
            if not os.path.exists(image_path): print(f"Warning: file does not exist: {image_path}"); return
            image = Image.open(image_path).convert("RGB")
            # Ensure the preprocessed image is a tensor
            preprocessed = self.clip_preprocess(image)
            if not isinstance(preprocessed, torch.Tensor):
                raise ValueError("CLIP preprocessing did not return a tensor")
            
            image_input = preprocessed.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image_input)
            
            return embedding.cpu().numpy()
        except Exception as e:
            print(f"Error in CLIP extraction: {e}")
            return None

    def _analyze_with_qwen_enhanced(self, image_path, yolo_objects, blip_caption, ocr_text, audio_text):
        if self.qwen_model is None or self.qwen_tokenizer is None:
            print("DEBUG: Qwen model or tokenizer is None, fallback.")
            return self._create_fallback_result(image_path, ocr_text, audio_text)
        try:
            prompt = "Describe the picture shortly"
            query = self.qwen_tokenizer.from_list_format([
                {'image': image_path},
                {'text': prompt}
            ])
            response, history = self.qwen_model.chat(self.qwen_tokenizer, query=query, history=None, max_new_tokens=64)
            description = response.strip() if response else "No description available"

            # Other fields use fallback
            fallback = self._create_fallback_result(image_path, ocr_text, audio_text)
            fallback['description'] = description
            return fallback

            # --- Original multi-field Qwen inference logic commented out ---
            # context_parts = []
            # if blip_caption:
            #     context_parts.append(f"BLIP caption: {blip_caption}")
            # if yolo_objects:
            #     context_parts.append(f"YOLO objects: {', '.join([obj['class'] for obj in yolo_objects[:3]])}")
            # if ocr_text:
            #     context_parts.append(f"Text in image: {ocr_text}")
            # if audio_text:
            #     context_parts.append(f"Audio transcription: {audio_text}")
            # context = " | ".join(context_parts)
            # query = self.qwen_tokenizer.from_list_format([
            #     {'image': image_path},
            #     {'text': context}
            # ])
            # response, history = self.qwen_model.chat(self.qwen_tokenizer, query=query, history=None)
            # if not response or response.strip() == '' or response.strip() in context or response.strip()[:100] in context:
            #     print("[DEBUG] Qwen response is empty or just prompt repeat! Possible model/weight/input issue.")
            # result = self._parse_qwen_response(response)
            # return result
        except Exception as e:
            print("ERROR in enhanced Qwen analysis:", e)
            return self._create_fallback_result(image_path, ocr_text, audio_text)

    def _parse_qwen_response(self, response):
        """Parse Qwen response with improved JSON extraction."""
        try:
            # Tolerantly extract the first JSON content wrapped in braces
            match = re.search(r'\{[\s\S]*?\}', response)
            if match:
                json_str = match.group(0)
                # Clean JSON string
                json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                # Parse JSON
                result = json.loads(json_str)
                
                # Check for placeholder text and replace with meaningful defaults
                description = result.get('description', 'No description available')
                primary_category = result.get('primary_category', 'Unknown')
                secondary_category = result.get('secondary_category', 'Unknown')
                tertiary_category = result.get('tertiary_category', 'Unknown')
                
                # Check for placeholder text and try to extract meaningful content
                placeholder_detected = False
                
                if 'detailed description of what you see' in description or 'detailed visual description' in description:
                    print("Warning: Detected placeholder text in description")
                    placeholder_detected = True
                    # Try to extract actual description from the response
                    description = self._extract_actual_description(response)
                
                if 'specific primary category' in primary_category or 'your specific primary category' in primary_category:
                    print("Warning: Detected placeholder text in primary_category")
                    placeholder_detected = True
                    primary_category = self._extract_actual_category(response, 'primary')
                
                if 'specific secondary category' in secondary_category or 'your specific secondary category' in secondary_category:
                    print("Warning: Detected placeholder text in secondary_category")
                    placeholder_detected = True
                    secondary_category = self._extract_actual_category(response, 'secondary')
                
                if 'specific tertiary category' in tertiary_category or 'your specific tertiary category' in tertiary_category:
                    print("Warning: Detected placeholder text in tertiary_category")
                    placeholder_detected = True
                    tertiary_category = self._extract_actual_category(response, 'tertiary')
                
                if placeholder_detected:
                    print("Attempting to extract actual content from Qwen response...")
                
                # Validate and clean results
                return {
                    'description': description,
                    'primary_category': primary_category,
                    'secondary_category': secondary_category,
                    'tertiary_category': tertiary_category,
                    'content_type': result.get('content_type', 'other'),
                    'target_audience': result.get('target_audience', 'general'),
                    'audio_relevance': result.get('audio_relevance', 'unknown'),
                    'audio_summary': result.get('audio_summary', 'none')
                }
            else:
                # Fallback: extract information from text
                print("Warning: No JSON found in response, using text extraction")
                return self._extract_info_from_text(response)
                
        except Exception as e:
            print(f"Error parsing Qwen response: {e}")
            print(f"Response was: {response[:200]}...")
            return self._extract_info_from_text(response)

    def _extract_actual_description(self, response):
        """Extract actual description from Qwen response."""
        try:
            # Look for description in the response text
            lines = response.split('\n')
            for line in lines:
                if '"description"' in line and ':' in line:
                    # Extract content between quotes
                    start = line.find('"description"') + len('"description"')
                    content = line[start:].strip()
                    if content.startswith(':'):
                        content = content[1:].strip()
                    if content.startswith('"'):
                        content = content[1:]
                    if content.endswith('"'):
                        content = content[:-1]
                    if content and not 'detailed description of what you see' in content:
                        return content
            
            # Fallback: look for any meaningful text
            for line in lines:
                if ':' in line and len(line) > 20 and not line.startswith('"'):
                    return line.strip()
            
            return "Image analysis completed"
        except:
            return "Image analysis completed"

    def _extract_actual_category(self, response, category_type):
        """Extract actual category from Qwen response."""
        try:
            lines = response.split('\n')
            for line in lines:
                if f'"{category_type}_category"' in line and ':' in line:
                    # Extract content between quotes
                    start = line.find(f'"{category_type}_category"') + len(f'"{category_type}_category"')
                    content = line[start:].strip()
                    if content.startswith(':'):
                        content = content[1:].strip()
                    if content.startswith('"'):
                        content = content[1:]
                    if content.endswith('"'):
                        content = content[:-1]
                    if content and not 'specific' in content.lower():
                        return content
            
            return "Unknown"
        except:
            return "Unknown"

    def _extract_info_from_text(self, text):
        """Extract information from text when JSON parsing fails."""
        # Simple text parsing as fallback
        lines = text.split('\n')
        description = ""
        primary = "Unknown"
        secondary = "Unknown"
        tertiary = "Unknown"
        
        for line in lines:
            line_lower = line.lower()
            if 'description' in line_lower and ':' in line:
                description = line.split(':', 1)[1].strip().strip('"')
            elif 'primary' in line_lower and 'category' in line_lower and ':' in line:
                primary = line.split(':', 1)[1].strip().strip('"')
            elif 'secondary' in line_lower and 'category' in line_lower and ':' in line:
                secondary = line.split(':', 1)[1].strip().strip('"')
            elif 'tertiary' in line_lower and 'category' in line_lower and ':' in line:
                tertiary = line.split(':', 1)[1].strip().strip('"')
        
        if not description:
            description = text[:200] + "..." if len(text) > 200 else text
        
        return {
            'description': description,
            'primary_category': primary,
            'secondary_category': secondary,
            'tertiary_category': tertiary,
            'content_type': 'other',
            'target_audience': 'general',
            'audio_relevance': 'unknown',
            'audio_summary': 'none'
        }

    def _create_fallback_result(self, image_path, ocr_text, audio_text):
        """Create fallback result when Qwen is not available."""
        try:
            # Use BLIP and YOLO for fallback
            if not os.path.exists(image_path): print(f"Warning: file does not exist: {image_path}"); return
            blip_caption = self._extract_blip_caption(image_path)
            yolo_objects = self._extract_yolo_features(image_path)
            
            description_parts = []
            if blip_caption:
                description_parts.append(f"BLIP: {blip_caption}")
            
            if yolo_objects:
                objects = [obj['class'] for obj in yolo_objects[:3]]
                description_parts.append(f"Objects: {', '.join(objects)}")
            
            if ocr_text:
                description_parts.append(f"OCR: {ocr_text}")
            
            if audio_text:
                description_parts.append(f"Audio: {audio_text}")
            
            description = " | ".join(description_parts) if description_parts else "No description available"
            
            # Simple categorization based on content
            primary, secondary, tertiary = self._simple_categorization(description, yolo_objects)
            
            return {
                'description': description,
                'primary_category': primary,
                'secondary_category': secondary,
                'tertiary_category': tertiary,
                'content_type': 'other',
                'target_audience': 'general',
                'audio_relevance': 'unknown',
                'audio_summary': 'none'
            }
            
        except Exception as e:
            print(f"Error in fallback result creation: {e}")
            return {
                'description': 'Analysis failed',
                'primary_category': 'Unknown',
                'secondary_category': 'Unknown',
                'tertiary_category': 'Unknown',
                'content_type': 'other',
                'target_audience': 'general',
                'audio_relevance': 'unknown',
                'audio_summary': 'none'
            }

    def _simple_categorization(self, description, yolo_objects):
        """Simple categorization based on keywords and detected objects."""
        description_lower = description.lower()
        object_classes = [obj['class'].lower() for obj in yolo_objects]
        
        # Enhanced keyword matching with more specific categories
        categories = {
            'Beauty and Personal Care': {
                'keywords': ['makeup', 'cosmetic', 'lipstick', 'foundation', 'skincare', 'beauty', 'perfume', 'shampoo'],
                'secondary': 'Makeup',
                'tertiary': 'Foundation'
            },
            'Fashion': {
                'keywords': ['clothing', 'dress', 'shirt', 'pants', 'shoes', 'jewelry', 'accessory', 'bag', 'watch', 'top', 'jeans', 'black'],
                'secondary': 'Clothing',
                'tertiary': 'Dresses'
            },
            'Electronics': {
                'keywords': ['phone', 'laptop', 'computer', 'camera', 'tv', 'electronic', 'device'],
                'secondary': 'Mobile Devices',
                'tertiary': 'Smartphones'
            },
            'Food and Beverages': {
                'keywords': ['food', 'drink', 'snack', 'beverage', 'meal', 'grocery', 'store', 'aisle'],
                'secondary': 'Snacks',
                'tertiary': 'Other'
            },
            'Home and Garden': {
                'keywords': ['furniture', 'home', 'kitchen', 'appliance', 'decoration', 'shelves', 'boxes'],
                'secondary': 'Furniture',
                'tertiary': 'Other'
            }
        }
        
        # Check each category with scoring
        best_score = 0
        best_category = ('Other', 'Other', 'Other')
        
        for primary, info in categories.items():
            score = 0
            # Check description keywords
            for keyword in info['keywords']:
                if keyword in description_lower:
                    score += 1
            
            # Check detected objects
            for keyword in info['keywords']:
                if keyword in object_classes:
                    score += 2  # Objects get higher weight
            
            if score > best_score:
                best_score = score
                best_category = (primary, info['secondary'], info['tertiary'])
        
        return best_category

    def extract_qwen_features(self, frame_dir, video_name=None, audio_transcript=""):
        """Extract features for all representative frames in a directory using enhanced analysis."""
        results = []
        frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.jpg') or f.endswith('.png')])
        for frame_path in frame_files:
            ocr_text = ""  # 可根据需要扩展 OCR
            analysis = self.extract_comprehensive_features(
                frame_path, ocr_text, audio_transcript
            )
            results.append({
                "frame": os.path.basename(frame_path),
                "frame_path": frame_path,
                "description": analysis['description'],
                "primary_category": analysis['primary_category'],
                "secondary_category": analysis['secondary_category'],
                "tertiary_category": analysis['tertiary_category'],
                "content_type": analysis['content_type'],
                "target_audience": analysis['target_audience'],
                "audio_relevance": analysis.get('audio_relevance', 'unknown'),
                "audio_summary": analysis.get('audio_summary', 'none'),
                "ocr_text": ocr_text
            })
        if not results:
            return {
                'video_description': '',
                'primary_category': 'Unknown',
                'secondary_category': 'Unknown',
                'tertiary_category': 'Unknown'
            }
        descriptions = [r['description'] for r in results]
        categories = {}
        for result in results:
            for key in ['primary_category', 'secondary_category', 'tertiary_category']:
                if key not in categories:
                    categories[key] = {}
                cat = result[key]
                categories[key][cat] = categories[key].get(cat, 0) + 1
        most_common = {}
        for key in categories:
            most_common[key] = max(categories[key].items(), key=lambda x: x[1])[0]
        summary_description = " | ".join(descriptions[:3])
        return {
            'video_description': summary_description,
            'primary_category': most_common['primary_category'],
            'secondary_category': most_common['secondary_category'],
            'tertiary_category': most_common['tertiary_category']
        }

    def generate_video_summary_csv(self, qwen_results, output_path):
        """Generate CSV with video description and AI categorization"""
        if not qwen_results:
            return
        
        # Aggregate results
        descriptions = []
        primary_categories = {}
        secondary_categories = {}
        tertiary_categories = {}
        
        for result in qwen_results:
            descriptions.append(result['description'])
            
            # Count categories
            primary = result['primary_category']
            secondary = result['secondary_category']
            tertiary = result['tertiary_category']
            
            primary_categories[primary] = primary_categories.get(primary, 0) + 1
            secondary_categories[secondary] = secondary_categories.get(secondary, 0) + 1
            tertiary_categories[tertiary] = tertiary_categories.get(tertiary, 0) + 1
        
        # Get most common categories
        most_common_primary = max(primary_categories.items(), key=lambda x: x[1])[0] if primary_categories else "Unknown"
        most_common_secondary = max(secondary_categories.items(), key=lambda x: x[1])[0] if secondary_categories else "Unknown"
        most_common_tertiary = max(tertiary_categories.items(), key=lambda x: x[1])[0] if tertiary_categories else "Unknown"
        
        # Create summary description
        summary_description = " | ".join(descriptions[:3])  # Take first 3 descriptions
        
        # Create DataFrame
        df = pd.DataFrame([{
            'Video_Name': qwen_results[0]['frame'].split('_representative_')[0] if qwen_results else "Unknown",
            'Description_of_Video': summary_description,
            'Primary_Category': most_common_primary,
            'Secondary_Category': most_common_secondary,
            'Tertiary_Category': most_common_tertiary,
            'Category_Confidence_Primary': primary_categories.get(most_common_primary, 0) / len(qwen_results),
            'Category_Confidence_Secondary': secondary_categories.get(most_common_secondary, 0) / len(qwen_results),
            'Category_Confidence_Tertiary': tertiary_categories.get(most_common_tertiary, 0) / len(qwen_results)
        }])
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Video summary saved to: {output_path}")
        
        return df 