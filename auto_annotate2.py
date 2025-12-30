import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class YOLOStackTracker:
    def __init__(self, video_path, model_path="stack.pt", conf_threshold=0.5, iou_threshold=0.5):
        """
        Initialize tracker with fine-tuned YOLO model
        
        Args:
            video_path: Path to input video
            model_path: Path to fine-tuned YOLO model (stack.pt)
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IoU threshold for NMS
        """
        self.video_path = video_path
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.tracked_objects = {}
        self.next_id = 1
        
    def load_model(self):
        """Load the fine-tuned YOLO model"""
        print(f"🔧 Loading fine-tuned model: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            print("✅ Model loaded successfully!")
            print(f"   Model classes: {self.model.names}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        
        # Union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def match_detections_to_tracks(self, detections, max_distance=50):
        """
        Match current detections to existing tracked objects
        
        Args:
            detections: List of current frame detections
            max_distance: Maximum distance for matching
        """
        matched_ids = set()
        unmatched_detections = []
        
        for det in detections:
            det_box = det['bbox']
            det_center = ((det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2)
            
            best_match_id = None
            best_iou = 0
            
            # Try to match with existing tracks
            for track_id, track_info in self.tracked_objects.items():
                if track_id in matched_ids:
                    continue
                
                track_box = track_info['bbox']
                iou = self.calculate_iou(det_box, track_box)
                
                if iou > 0.3 and iou > best_iou:  # IoU threshold for matching
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                self.tracked_objects[best_match_id]['bbox'] = det_box
                self.tracked_objects[best_match_id]['class'] = det['class']
                self.tracked_objects[best_match_id]['confidence'] = det['confidence']
                self.tracked_objects[best_match_id]['lost_frames'] = 0
                matched_ids.add(best_match_id)
                det['id'] = best_match_id
            else:
                # New detection
                unmatched_detections.append(det)
        
        # Add new tracks for unmatched detections
        for det in unmatched_detections:
            self.tracked_objects[self.next_id] = {
                'bbox': det['bbox'],
                'class': det['class'],
                'confidence': det['confidence'],
                'lost_frames': 0
            }
            det['id'] = self.next_id
            self.next_id += 1
        
        # Remove tracks that have been lost for too long
        lost_threshold = 30  # frames
        tracks_to_remove = []
        for track_id, track_info in self.tracked_objects.items():
            if track_id not in matched_ids:
                track_info['lost_frames'] += 1
                if track_info['lost_frames'] > lost_threshold:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
    
    def detect_and_track(self, output_path=None, display_size=(1280, 720)):
        """
        Run detection and tracking on video
        
        Args:
            output_path: Optional path to save output video
            display_size: Size for display window (width, height)
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("❌ Error opening video file")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n📹 Video Info:")
        print(f"   Frames: {total_frames}")
        print(f"   FPS: {fps}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Display size: {display_size[0]}x{display_size[1]}")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, display_size)
            print(f"💾 Saving output to: {output_path}")
        
        print("\n🎬 Starting detection and tracking...")
        print("   Press 'q' to quit")
        print("   Press 'p' to pause/resume")
        print("   Press 's' to save current frame")
        
        frame_count = 0
        paused = False
        
        # Color palette for different classes
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Resize frame for display
                display_frame = cv2.resize(frame, display_size)
                
                # Run YOLO detection
                results = self.model(display_frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
                
                # Extract detections
                detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[cls]
                        
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': conf,
                            'class': class_name,
                            'class_id': cls
                        })
                
                # Match detections to tracks
                self.match_detections_to_tracks(detections)
                
                # Draw detections with tracking IDs
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    color = colors[det.get('id', 0) % len(colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label
                    label = f"ID:{det.get('id', '?')} {det['class']} {det['confidence']:.2f}"
                    
                    # Draw label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(display_frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw info panel
                info_bg = display_frame.copy()
                cv2.rectangle(info_bg, (0, 0), (400, 100), (0, 0, 0), -1)
                display_frame = cv2.addWeighted(display_frame, 0.7, info_bg, 0.3, 0)
                
                cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Detected: {len(detections)} stacks", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Tracked: {len(self.tracked_objects)} objects", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Write frame if output video specified
                if writer:
                    writer.write(display_frame)
            
            # Display frame
            cv2.imshow("YOLO Stack Tracking", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
            elif key == ord('s'):
                screenshot_path = f"frame_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, display_frame)
                print(f"📸 Saved frame to {screenshot_path}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Processing complete!")
        print(f"   Total frames processed: {frame_count}")
        print(f"   Final tracked objects: {len(self.tracked_objects)}")
    
    def run(self, output_path=None, display_size=(1280, 720)):
        """Main execution"""
        print("🚀 YOLO Stack Tracker with Fine-tuned Model")
        print("=" * 60)
        
        # Load model
        self.load_model()
        
        # Run detection and tracking
        self.detect_and_track(output_path=output_path, display_size=display_size)


# ============= USAGE =============
if __name__ == "__main__":
    # Basic usage
    tracker = YOLOStackTracker(
        video_path="Screen Recording 2025-12-30 183422.mp4", #add your video path
        model_path="stack.pt",
        conf_threshold=0.5,  # Adjust confidence threshold as needed
        iou_threshold=0.5    # Adjust IoU threshold for NMS
    )
    
    # Run WITH output video saved (annotated video will be downloaded)
    tracker.run(output_path="tracked_output2.mp4", display_size=(1280, 720))
    
    # Or run without saving output
    # tracker.run()
    
    # You can also adjust display size for better visualization
    # tracker.run(display_size=(1920, 1080))  # Full HD
    # tracker.run(display_size=(640, 640))    # Square format
