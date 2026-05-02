#!/usr/bin/env python3
"""
Bottle Detection & Damage Classification Pipeline for Raspberry Pi 5
=====================================================================

Pipeline:
1. YOLO detects bottles → draws bounding boxes
2. Crops each detected bottle
3. ResNet50 TFLite classifies each crop (damaged / not_damaged)
4. Logs results: total bottles, damaged count, individual results

Required files:
- yolov8n.pt (or fine-tuned YOLO model)
- bottle_classifier_resnet50v2_int8.tflite
- labels.txt

Usage:
    python bottle_pipeline.py --image photo.jpg
    python bottle_pipeline.py --folder images/
    python bottle_pipeline.py --camera
"""

import os
import cv2
import numpy as np
import argparse
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BottleDetection:
    """Single bottle detection result."""
    bottle_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    detection_confidence: float
    damage_class: str  # 'damaged' or 'non_damaged'
    damage_confidence: float
    crop_path: Optional[str] = None


@dataclass
class PipelineResult:
    """Complete pipeline result for one image."""
    image_path: str
    timestamp: str
    total_bottles: int
    damaged_count: int
    not_damaged_count: int
    detection_time_ms: float
    classification_time_ms: float
    total_time_ms: float
    detections: List[BottleDetection]


def remove_overlapping_boxes(detections, iou_threshold=0.3, containment_threshold=0.8):
    """
    Remove duplicate bounding boxes while preserving multiple bottles.
    
    Logic:
    - If a large box contains 1 or more smaller boxes (>80% inside), 
      the large box is a duplicate → remove the large box, keep small ones
    - This handles both:
      - Single bottle with duplicate boxes (1 small inside 1 large)
      - Multiple bottles where YOLO drew one big box around all of them
    
    Args:
        detections: List of (x1, y1, x2, y2, confidence) tuples
        iou_threshold: IoU threshold for overlap detection
        containment_threshold: How much of small box must be inside large box
    
    Returns:
        Filtered list of detections (individual bottles only)
    """
    if len(detections) <= 1:
        return detections
    
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    
    def get_intersection_area(box1, box2):
        """Calculate intersection area between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)
    
    def is_contained(small_box, large_box):
        """Check if small_box is mostly contained within large_box."""
        inter_area = get_intersection_area(small_box, large_box)
        small_area = box_area(small_box)
        
        if small_area == 0:
            return False
        
        return (inter_area / small_area) >= containment_threshold
    
    # Sort by area (largest first for checking containment)
    sorted_dets = sorted(detections, key=lambda x: box_area(x), reverse=True)
    
    boxes_to_remove = set()
    
    for i, large_box in enumerate(sorted_dets):
        if i in boxes_to_remove:
            continue
            
        # Count how many smaller boxes are contained in this large box
        contained_count = 0
        
        for j, small_box in enumerate(sorted_dets):
            if i == j or j in boxes_to_remove:
                continue
            
            # Check if small_box is inside large_box
            if box_area(small_box) < box_area(large_box):
                if is_contained(small_box, large_box):
                    contained_count += 1
        
        # If 1+ smaller boxes inside → large box is duplicate/wrapper, remove it
        # This handles:
        #   - 1 small box inside = duplicate detection of same bottle
        #   - 2+ small boxes inside = large box spanning multiple bottles
        if contained_count >= 1:
            boxes_to_remove.add(i)
    
    # Return boxes that weren't marked for removal
    result = [
        box for idx, box in enumerate(sorted_dets)
        if idx not in boxes_to_remove
    ]
    
    return result


class BottleDamageClassifier:
    """
    TFLite-based damage classifier for cropped bottle images.
    Uses ResNet50V2 INT8 quantized model.
    """
    
    def __init__(self, model_path: str, labels_path: str = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to .tflite model file
            labels_path: Path to labels.txt (optional, defaults to ['damaged', 'non_damaged'])
        """
        self.model_path = model_path
        self.img_size = (224, 224)
        
        # Load labels
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            self.class_names = ['damaged', 'non_damaged']
        
        logger.info(f"Classes: {self.class_names}")
        
        # Load TFLite model
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=model_path)
            logger.info("Using tflite_runtime")
        except ImportError:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            logger.info("Using tensorflow.lite")
        
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        
        logger.info(f"Damage classifier loaded: {model_path}")
        logger.info(f"Input shape: {self.input_details['shape']}")
        logger.info(f"Input dtype: {self.input_details['dtype']}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for TFLite model.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Preprocessed image array
        """
        # Resize
        img = cv2.resize(image, self.img_size)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Handle quantized model (INT8)
        if self.input_details['dtype'] == np.uint8:
            img = img.astype(np.uint8)
        else:
            # Float model - ResNet50V2 preprocessing: scale to [-1, 1]
            # This matches tf.keras.applications.resnet_v2.preprocess_input()
            img = img.astype(np.float32)
            img = (img / 127.5) - 1.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Classify a cropped bottle image.
        
        Args:
            image: BGR image of cropped bottle
            
        Returns:
            Tuple of (class_name, confidence)
        """
        # Preprocess
        input_data = self.preprocess(image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details['index'], input_data)
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details['index'])[0]
        
        # Handle quantized output
        if self.output_details['dtype'] == np.int8 or self.output_details['dtype'] == np.uint8:
            scale, zero_point = self.output_details['quantization']
            output = scale * (output.astype(np.float32) - zero_point)
        
        # Softmax if needed (some models don't have it)
        if len(output) > 1:
            exp_output = np.exp(output - np.max(output))
            probs = exp_output / np.sum(exp_output)
            
            # --- ADDED PROBABILITY LOGGING HERE ---
            prob_dict = {self.class_names[i]: float(probs[i]) for i in range(len(probs))}
            logger.info(f"Raw Probabilities: {prob_dict}")
            # --------------------------------------
        else:
            probs = output
            logger.info(f"Raw Probability: {probs}")
        
        # Get prediction
        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx])
        class_name = self.class_names[class_idx]
        
        return class_name, confidence


class BottleDetectionPipeline:
    """
    Complete pipeline: YOLO detection + ResNet damage classification.
    """
    
    def __init__(
        self,
        yolo_model_path: str = 'yolov8n.pt',
        classifier_model_path: str = 'bottle_classifier_resnet50v2_int8.tflite',
        labels_path: str = 'labels.txt',
        detection_confidence: float = 0.10,
        crop_padding: int = 20,
        output_dir: str = 'pipeline_output'
    ):
        """
        Initialize the pipeline.
        
        Args:
            yolo_model_path: Path to YOLO model (.pt file)
            classifier_model_path: Path to TFLite damage classifier
            labels_path: Path to labels.txt
            detection_confidence: YOLO detection confidence threshold
            crop_padding: Padding around detected bottles (pixels)
            output_dir: Directory to save results
        """
        self.detection_confidence = detection_confidence
        self.crop_padding = crop_padding
        self.output_dir = output_dir
        
        # Create output directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/crops").mkdir(exist_ok=True)
        Path(f"{output_dir}/annotated").mkdir(exist_ok=True)
        
        # Load YOLO model
        logger.info(f"Loading YOLO model: {yolo_model_path}")
        from ultralytics import YOLO
        self.yolo = YOLO(yolo_model_path)
        logger.info("YOLO model loaded")
        
        # Load damage classifier
        logger.info(f"Loading damage classifier: {classifier_model_path}")
        self.classifier = BottleDamageClassifier(classifier_model_path, labels_path)
        logger.info("Damage classifier loaded")
        
        # Results log
        self.results_log = []
    
    def detect_bottles(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect bottles using YOLO.
        Applies custom NMS to remove duplicate/overlapping boxes.
        
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        results = self.yolo(image, conf=self.detection_confidence, verbose=False)
        
        raw_detections = []
        for result in results:
            for box in result.boxes:
                cls_name = result.names[int(box.cls[0])]
                if cls_name == 'bottle':
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    raw_detections.append((x1, y1, x2, y2, conf))
        
        # Remove overlapping boxes - keep tighter (smaller) ones
        detections = remove_overlapping_boxes(raw_detections, iou_threshold=0.3)
        
        logger.debug(f"Raw detections: {len(raw_detections)}, After NMS: {len(detections)}")
        
        return detections
    
    def crop_bottle(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Crop bottle region from image with padding.
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - self.crop_padding)
        y1 = max(0, y1 - self.crop_padding)
        x2 = min(w, x2 + self.crop_padding)
        y2 = min(h, y2 + self.crop_padding)
        
        return image[y1:y2, x1:x2].copy()
    
    def process_image(
        self,
        image_path: str,
        save_crops: bool = True,
        save_annotated: bool = True
    ) -> PipelineResult:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to input image
            save_crops: Save cropped bottle images
            save_annotated: Save annotated image with boxes and labels
            
        Returns:
            PipelineResult with all detection and classification info
        """
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        h, w = image.shape[:2]
        image_name = Path(image_path).stem
        
        # Step 1: YOLO Detection
        detection_start = time.time()
        detections = self.detect_bottles(image)
        detection_time = (time.time() - detection_start) * 1000
        
        logger.info(f"Detected {len(detections)} bottle(s) in {detection_time:.1f}ms")
        
        # Step 2: Classify each detection
        classification_start = time.time()
        
        annotated_image = image.copy()
        bottle_results = []
        damaged_count = 0
        not_damaged_count = 0
        
        for idx, (x1, y1, x2, y2, det_conf) in enumerate(detections):
            # Crop bottle
            crop = self.crop_bottle(image, (x1, y1, x2, y2))
            
            # Classify damage
            damage_class, damage_conf = self.classifier.predict(crop)
            
            # Count
            if damage_class == 'damaged':
                damaged_count += 1
                box_color = (0, 0, 255)  # Red for damaged
            else:
                not_damaged_count += 1
                box_color = (0, 255, 0)  # Green for not damaged
            
            # Save crop if requested
            crop_path = None
            if save_crops:
                crop_filename = f"{image_name}_bottle_{idx}_{damage_class}.jpg"
                crop_path = f"{self.output_dir}/crops/{crop_filename}"
                cv2.imwrite(crop_path, crop)
            
            # Draw on annotated image
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, 3)
            
            label = f"{damage_class}: {damage_conf:.0%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 5, y1),
                box_color, -1
            )
            cv2.putText(
                annotated_image, label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            # Store result
            bottle_results.append(BottleDetection(
                bottle_id=idx,
                bbox=(x1, y1, x2, y2),
                detection_confidence=det_conf,
                damage_class=damage_class,
                damage_confidence=damage_conf,
                crop_path=crop_path
            ))
        
        classification_time = (time.time() - classification_start) * 1000
        total_time = (time.time() - start_time) * 1000
        
        # Save annotated image
        if save_annotated:
            annotated_path = f"{self.output_dir}/annotated/{image_name}_result.jpg"
            
            # Add summary text
            summary = f"Total: {len(detections)} | Damaged: {damaged_count} | OK: {not_damaged_count}"
            cv2.putText(
                annotated_image, summary,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            
            cv2.imwrite(annotated_path, annotated_image)
            logger.info(f"Saved annotated image: {annotated_path}")
        
        # Create result
        result = PipelineResult(
            image_path=image_path,
            timestamp=datetime.now().isoformat(),
            total_bottles=len(detections),
            damaged_count=damaged_count,
            not_damaged_count=not_damaged_count,
            detection_time_ms=round(detection_time, 2),
            classification_time_ms=round(classification_time, 2),
            total_time_ms=round(total_time, 2),
            detections=[asdict(d) for d in bottle_results]
        )
        
        # Log result
        self.results_log.append(asdict(result))
        
        return result
    
    def process_folder(self, folder_path: str) -> List[PipelineResult]:
        """
        Process all images in a folder.
        """
        folder = Path(folder_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
        
        logger.info(f"Processing {len(images)} images from {folder_path}")
        
        results = []
        for img_path in images:
            try:
                result = self.process_image(str(img_path))
                results.append(result)
                
                logger.info(
                    f"{img_path.name}: {result.total_bottles} bottles, "
                    f"{result.damaged_count} damaged, {result.total_time_ms:.0f}ms"
                )
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
        
        return result
    
    
    
    def process_camera(
        self,
        camera_id: int = 0,
        display: bool = True,
        save_detections: bool = False
    ):
        """
        Photo Booth Workflow:
        1. Live Countdown (5s) -> 2. Capture & Process -> 3. Show Result (10s) -> Repeat
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.error("Could not open camera")
            return
        
        logger.info("Starting Photo Booth Mode. Press 'q' at any time to quit.")
        
        frame_count = 0
        total_bottles = 0
        total_damaged = 0
        
        try:
            while True:
                # ==========================================
                # PHASE 1: LIVE COUNTDOWN (5 SECONDS)
                # ==========================================
                logger.info("Position the bottle. Capturing in 5 seconds...")
                countdown_start = time.time()
                
                # Keep pulling frames for 5 seconds to show a live feed and clear buffer lag
                while time.time() - countdown_start < 5.0:
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Camera disconnected.")
                        return
                    
                    time_left = int(5.0 - (time.time() - countdown_start)) + 1
                    
                    if display:
                        view = frame.copy()
                        cv2.putText(view, f"Capturing in: {time_left}", (10, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        cv2.imshow('Bottle Damage Detection', view)
                        
                        # A short 30ms wait keeps the video feed smooth and responsive
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord('q'):
                            return # Exit completely
                
                # ==========================================
                # PHASE 2: CAPTURE & PROCESS
                # ==========================================
                # The 'frame' variable now holds the final image taken at 0 seconds
                if display:
                    processing_view = frame.copy()
                    cv2.putText(processing_view, "Processing AI...", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                    cv2.imshow('Bottle Damage Detection', processing_view)
                    # We wait 100ms here to FORCE OpenCV to draw the screen before the AI hogs the CPU
                    cv2.waitKey(100) 
                
                start_time = time.time()
                annotated = frame.copy()
                
                # Run Inference
                detections = self.detect_bottles(frame)
                
                frame_damaged = 0
                for x1, y1, x2, y2, det_conf in detections:
                    crop = self.crop_bottle(frame, (x1, y1, x2, y2))
                    damage_class, damage_conf = self.classifier.predict(crop)
                    
                    if damage_class == 'damaged':
                        frame_damaged += 1
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                    
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{damage_class}: {damage_conf:.0%}"
                    cv2.putText(annotated, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Update totals
                total_bottles += len(detections)
                total_damaged += frame_damaged
                process_time = time.time() - start_time
                
                # Draw Info
                info = [
                    f"Process Time: {process_time:.2f}s",
                    f"Frame bottles: {len(detections)}",
                    f"Frame damaged: {frame_damaged}",
                    f"Total bottles: {total_bottles}",
                    f"Total damaged: {total_damaged}"
                ]
                
                y = 30
                for line in info:
                    cv2.putText(annotated, line, (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y += 25
                
                # ==========================================
                # PHASE 3: SHOW RESULTS FOR 10 SECONDS
                # ==========================================
                if display:
                    cv2.imshow('Bottle Damage Detection', annotated)
                    logger.info(f"Analyzed {len(detections)} bottles in {process_time:.2f}s. Displaying for 10s...")
                    
                    result_start = time.time()
                    
                    # A dedicated loop keeps the window alive and responsive for exactly 10 seconds
                    while time.time() - result_start < 10.0:
                        key = cv2.waitKey(100) & 0xFF
                        if key == ord('q'):
                            return
                        elif key == ord('s'):
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            cv2.imwrite(f"{self.output_dir}/camera_{timestamp}.jpg", annotated)
                            logger.info(f"Saved frame manually.")
                
                frame_count += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            logger.info(f"\n{'='*50}")
            logger.info("SESSION SUMMARY")
            logger.info(f"{'='*50}")
            logger.info(f"Snapshots processed: {frame_count}")
            logger.info(f"Total bottles detected: {total_bottles}")
            logger.info(f"Total damaged: {total_damaged}")
            if total_bottles > 0:
                logger.info(f"Damage rate: {total_damaged/total_bottles*100:.1f}%")



    def save_results(self, output_path: str = None):
        """
        Save all results to JSON file.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.output_dir}/results_{timestamp}.json"
        
        # Calculate summary
        total_images = len(self.results_log)
        total_bottles = sum(r['total_bottles'] for r in self.results_log)
        total_damaged = sum(r['damaged_count'] for r in self.results_log)
        total_ok = sum(r['not_damaged_count'] for r in self.results_log)
        
        output = {
            'summary': {
                'total_images': total_images,
                'total_bottles': total_bottles,
                'total_damaged': total_damaged,
                'total_not_damaged': total_ok,
                'damage_rate': f"{total_damaged/total_bottles*100:.1f}%" if total_bottles > 0 else "N/A"
            },
            'results': self.results_log
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print summary of all processed images."""
        total_images = len(self.results_log)
        total_bottles = sum(r['total_bottles'] for r in self.results_log)
        total_damaged = sum(r['damaged_count'] for r in self.results_log)
        total_ok = sum(r['not_damaged_count'] for r in self.results_log)
        avg_time = np.mean([r['total_time_ms'] for r in self.results_log]) if self.results_log else 0
        
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"  Images processed:    {total_images}")
        print(f"  Total bottles:       {total_bottles}")
        print(f"  Damaged:             {total_damaged}")
        print(f"  Not damaged:         {total_ok}")
        if total_bottles > 0:
            print(f"  Damage rate:         {total_damaged/total_bottles*100:.1f}%")
        print(f"  Avg processing time: {avg_time:.0f}ms per image")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Bottle Detection & Damage Classification Pipeline'
    )
    parser.add_argument(
        '--mode', type=str, default='image',
        choices=['image', 'folder', 'camera'],
        help='Processing mode'
    )
    parser.add_argument(
        '--input', '-i', type=str,
        help='Input image path or folder'
    )
    parser.add_argument(
        '--yolo-model', type=str, default='yolov8s.pt',
        help='Path to YOLO model'
    )
    parser.add_argument(
        '--classifier-model', type=str,
        default='bottle_classifier_resnet50v2_int8.tflite',
        help='Path to TFLite damage classifier'
    )
    parser.add_argument(
        '--labels', type=str, default='labels.txt',
        help='Path to labels.txt'
    )
    parser.add_argument(
        '--confidence', '-c', type=float, default=0.10,
        help='YOLO detection confidence'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='pipeline_output',
        help='Output directory'
    )
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera device ID'
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='Disable display for camera mode'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = BottleDetectionPipeline(
        yolo_model_path=args.yolo_model,
        classifier_model_path=args.classifier_model,
        labels_path=args.labels,
        detection_confidence=args.confidence,
        output_dir=args.output
    )
    
    if args.mode == 'image':
        if not args.input:
            print("Error: --input required for image mode")
            return
        
        result = pipeline.process_image(args.input)
        
        print(f"\n📷 Image: {args.input}")
        print(f"   Bottles detected: {result.total_bottles}")
        print(f"   Damaged: {result.damaged_count}")
        print(f"   Not damaged: {result.not_damaged_count}")
        print(f"   Time: {result.total_time_ms:.0f}ms")
        
        for det in result.detections:
            print(f"\n   Bottle #{det['bottle_id']}:")
            print(f"     Detection conf: {det['detection_confidence']:.0%}")
            print(f"     Damage class: {det['damage_class']}")
            print(f"     Damage conf: {det['damage_confidence']:.0%}")
    
    elif args.mode == 'folder':
        if not args.input:
            print("Error: --input required for folder mode")
            return
        
        results = pipeline.process_folder(args.input)
        pipeline.print_summary()
        pipeline.save_results()
    
    elif args.mode == 'camera':
        pipeline.process_camera(
            camera_id=args.camera,
            display=not args.no_display
        )
        pipeline.save_results()


if __name__ == "__main__":
    main()
