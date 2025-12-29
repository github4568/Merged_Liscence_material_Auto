import cv2
import numpy as np
from pathlib import Path

class DirectStackTracker:
    def __init__(self, video_path, num_examples=4):
        self.video_path = video_path
        self.num_examples = num_examples
        self.manual_labels = []
        self.drawing = False
        self.start_point = None
        self.current_frame = None
        self.temp_frame = None
        self.trackers = []
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for bounding box drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.temp_frame = self.current_frame.copy()
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_frame = self.current_frame.copy()
                cv2.rectangle(self.temp_frame, self.start_point, (x, y), (0, 255, 0), 2)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)
            
            # Store the bounding box
            bbox = {
                'x1': min(self.start_point[0], end_point[0]),
                'y1': min(self.start_point[1], end_point[1]),
                'x2': max(self.start_point[0], end_point[0]),
                'y2': max(self.start_point[1], end_point[1])
            }
            self.manual_labels[-1]['bboxes'].append(bbox)
            
            # Draw final box
            cv2.rectangle(self.current_frame, self.start_point, end_point, (0, 255, 0), 2)
            cv2.putText(self.current_frame, "Stack", self.start_point, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def extract_sample_frames(self):
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🔍 Video has {total_frames} frames")
        
        # Get evenly distributed frames
        frame_indices = np.linspace(0, total_frames-1, self.num_examples * 3, dtype=int)
        
        candidate_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                candidate_frames.append({
                    'frame_idx': idx,
                    'frame': frame.copy()
                })
        
        cap.release()
        return candidate_frames
    
    def manual_labeling_phase(self):
        """Interactive manual labeling window"""
        all_sample_frames = self.extract_sample_frames()
        
        print(f"\n📝 Manual Labeling Phase - Need {self.num_examples} labeled frames")
        print("Instructions:")
        print("  - Click and drag to draw bounding box around stack/material")
        print("  - Press SPACE to finish labeling current frame")
        print("  - Press ESC to skip frame")
        
        frame_pool_idx = 0
        labeled_count = 0
        
        while labeled_count < self.num_examples and frame_pool_idx < len(all_sample_frames):
            sample = all_sample_frames[frame_pool_idx]
            
            # Resize to 640x640
            original_frame = sample['frame']
            resized_frame = cv2.resize(original_frame, (640, 640))
            
            self.current_frame = resized_frame.copy()
            self.temp_frame = self.current_frame.copy()
            
            # Prepare to store labels
            temp_label = {
                'frame_idx': sample['frame_idx'],
                'bboxes': []
            }
            
            window_name = f"Label Frame {labeled_count+1}/{self.num_examples} (640x640)"
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, self.mouse_callback)
            
            label_idx = len(self.manual_labels)
            self.manual_labels.append(temp_label)
            
            print(f"\n📸 Frame {frame_pool_idx + 1} (need {self.num_examples - labeled_count} more)")
            
            while True:
                display_frame = self.temp_frame if self.drawing else self.current_frame
                
                cv2.putText(display_frame, f"Labeled: {labeled_count}/{self.num_examples}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, "SPACE: Done | ESC: Skip", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE
                    if len(self.manual_labels[label_idx]['bboxes']) > 0:
                        print(f"✅ Labeled with {len(self.manual_labels[label_idx]['bboxes'])} boxes")
                        labeled_count += 1
                    else:
                        print("⚠️ No boxes, skipping...")
                        self.manual_labels.pop(label_idx)
                    break
                elif key == 27:  # ESC
                    print("⏭️ Skipped")
                    self.manual_labels.pop(label_idx)
                    break
            
            cv2.destroyWindow(window_name)
            frame_pool_idx += 1
        
        print(f"\n✅ Got {len(self.manual_labels)} labeled frames!")
        self.show_understood_popup()
    
    def show_understood_popup(self):
        """Show confirmation"""
        popup = np.zeros((250, 500, 3), dtype=np.uint8)
        popup[:] = (40, 40, 40)
        
        cv2.putText(popup, "UNDERSTOOD!", (120, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(popup, f"Labeled {len(self.manual_labels)} frames", (80, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(popup, "Starting tracking...", (90, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(popup, "Press any key", (150, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Status", popup)
        cv2.waitKey(0)
        cv2.destroyWindow("Status")
    
    def initialize_trackers(self, frame):
        """Store templates from labeled boxes"""
        self.trackers = []
        
        for label in self.manual_labels:
            for bbox in label['bboxes']:
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                
                # Extract template from frame
                template = frame[y1:y2, x1:x2].copy()
                
                self.trackers.append({
                    'template': template,
                    'last_pos': (x1, y1, x2, y2),
                    'id': len(self.trackers) + 1,
                    'active': True
                })
        
        print(f"🎯 Initialized {len(self.trackers)} trackers")

    def track_and_display(self):
        """Play video with template matching tracking"""
        cap = cv2.VideoCapture(self.video_path)
        
        print("\n🎬 Playing video with tracked stacks...")
        print("Press 'q' to quit")
        
        # Find first labeled frame
        first_labeled_frame_idx = min([label['frame_idx'] for label in self.manual_labels])
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_labeled_frame_idx)
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Couldn't read video")
            return
        
        frame = cv2.resize(frame, (640, 640))
        
        # Initialize with templates
        self.initialize_trackers(frame)
        
        frame_count = first_labeled_frame_idx
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame = cv2.resize(frame, (640, 640))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Track each template
            active_count = 0
            for tracker_obj in self.trackers:
                if tracker_obj['active']:
                    template = cv2.cvtColor(tracker_obj['template'], cv2.COLOR_BGR2GRAY)
                    
                    # Get search region around last position
                    x1, y1, x2, y2 = tracker_obj['last_pos']
                    search_margin = 50
                    
                    sx1 = max(0, x1 - search_margin)
                    sy1 = max(0, y1 - search_margin)
                    sx2 = min(640, x2 + search_margin)
                    sy2 = min(640, y2 + search_margin)
                    
                    search_region = frame_gray[sy1:sy2, sx1:sx2]
                    
                    if search_region.size > 0 and template.size > 0:
                        # Template matching
                        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                        
                        if max_val > 0.5:  # Confidence threshold
                            # Update position
                            h, w = template.shape
                            new_x1 = sx1 + max_loc[0]
                            new_y1 = sy1 + max_loc[1]
                            new_x2 = new_x1 + w
                            new_y2 = new_y1 + h
                            
                            tracker_obj['last_pos'] = (new_x1, new_y1, new_x2, new_y2)
                            
                            # Draw box
                            cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)
                            label = f"Material #{tracker_obj['id']}"
                            cv2.putText(frame, label, (new_x1, new_y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            active_count += 1
                        else:
                            tracker_obj['active'] = False
            
            # Display info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracking: {active_count} stacks", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Material Tracking - 640x640", frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Complete!")
        
        def track_and_display(self):
            """Play video with tracking based on your labels"""
            cap = cv2.VideoCapture(self.video_path)
            
            print("\n🎬 Playing video with tracked stacks...")
            print("Press 'q' to quit, 'r' to reset trackers")
            
            # Find first labeled frame to initialize trackers
            first_labeled_frame_idx = min([label['frame_idx'] for label in self.manual_labels])
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_labeled_frame_idx)
            
            ret, frame = cap.read()
            if not ret:
                print("❌ Couldn't read video")
                return
            
            frame = cv2.resize(frame, (640, 640))
            
            # Initialize trackers with your labeled boxes
            self.initialize_trackers(frame)
            
            frame_count = first_labeled_frame_idx
            material_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                frame = cv2.resize(frame, (640, 640))
                
                # Update all trackers
                active_count = 0
                for i, tracker_obj in enumerate(self.trackers):
                    if tracker_obj['active']:
                        success, bbox = tracker_obj['tracker'].update(frame)
                        
                        if success:
                            # Draw tracked box
                            x, y, w, h = [int(v) for v in bbox]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            label = f"Material #{tracker_obj['id']}"
                            cv2.putText(frame, label, (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            active_count += 1
                        else:
                            # Tracking lost
                            tracker_obj['active'] = False
                
                material_count = active_count
                
                # Display info
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Tracking: {material_count} stacks", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow("Material Tracking - 640x640", frame)
                
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset trackers
                    self.initialize_trackers(frame)
                    print("🔄 Trackers reset")
            
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n✅ Complete!")
    
    def run(self):
        """Main execution"""
        print("🚀 Direct Stack Tracker (No Model)")
        print("=" * 50)
        
        # Phase 1: Label
        self.manual_labeling_phase()
        
        # Phase 2: Track
        self.track_and_display()


# ============= USAGE =============
if __name__ == "__main__":
    tracker = DirectStackTracker(
        video_path="auto_label (1).mp4",
        num_examples=4
    )
    
    tracker.run()