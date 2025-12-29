import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# Load your UNIFIED model with OBB (Oriented Bounding Box) support
model = YOLO("roll6.pt")

# ===============================
# PARAMETERS
# ===============================
display = True
DISAPPEAR_THRESHOLD = 20

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.5
CONFIDENCE_ROLLS = 0.5
CONFIDENCE_SIDE = 0.5
CONFIDENCE_TOP = 0.5

# Size filters
MIN_ROLL_WIDTH = 10
MIN_ROLL_HEIGHT = 10
MAX_ROLL_WIDTH = 200
MAX_ROLL_HEIGHT = 200
MIN_SIDE_AREA = 500
MIN_TOP_DEPTH = 10

# Rolling window size for median calculation
ROLLING_WINDOW_SIZE = 30

# Sanity check limits
MAX_REASONABLE_ROLLS = 500
MIN_REASONABLE_ROLLS = 1

# Dimension validation tolerance
DIMENSION_TOLERANCE_PIXELS = 25

# NEW: Choose dimension extraction method
# Options: "obb_direct", "rotated_rect_minmax", "contour_area"
DIMENSION_METHOD = "obb_direct"  # RECOMMENDED: Use OBB width/height directly

# ===============================
# KNOWN STACK PATTERNS
# ===============================
KNOWN_PATTERNS = [
    {'wide': 3, 'high': 8, 'depth': 2, 'total': 48, 'name': '3x8x2'},
    {'wide': 3, 'high': 7, 'depth': 2, 'total': 42, 'name': '3x7x2'},
    {'wide': 2, 'high': 8, 'depth': 2, 'total': 32, 'name': '2x8x2'},
    {'wide': 3, 'high': 8, 'depth': 3, 'total': 72, 'name': '3x8x3'},
    {'wide': 4, 'high': 8, 'depth': 2, 'total': 64, 'name': '4x8x2'},
]

# ===============================
# TRACKING VARIABLES
# ===============================
roll_widths = deque(maxlen=ROLLING_WINDOW_SIZE)
roll_heights = deque(maxlen=ROLLING_WINDOW_SIZE)

side_widths = deque(maxlen=ROLLING_WINDOW_SIZE)
side_heights = deque(maxlen=ROLLING_WINDOW_SIZE)
top_heights = deque(maxlen=ROLLING_WINDOW_SIZE)

median_roll_width = 0
median_roll_height = 0
median_roll_area = 0
median_side_width = 0
median_side_height = 0
median_side_area = 0
median_top_height = 0

stack_was_visible = False
stack_disappeared_frames = 0
stack_calculation_done = False
stack_first_seen_frame = 0

calculated_rolls_current_stack = 0
total_material_in_rolls = 0

total_frames_processed = 0
total_side_detections = 0
total_top_detections = 0
total_roll_detections = 0
total_corrections_applied = 0
total_pattern_matches = 0

def calculate_median_safe(data_deque):
    """Safely calculate median from deque"""
    if len(data_deque) == 0:
        return 0
    return int(np.median(list(data_deque)))

def validate_calculation(calculated_rolls, reason=""):
    """Sanity check for calculated rolls"""
    if calculated_rolls < MIN_REASONABLE_ROLLS:
        print(f"⚠️ WARNING: Calculated {calculated_rolls} rolls - too low! {reason}")
        return False
    if calculated_rolls > MAX_REASONABLE_ROLLS:
        print(f"⚠️ WARNING: Calculated {calculated_rolls} rolls - too high! {reason}")
        return False
    return True

def extract_obb_dimensions_method1(obb_box):
    """
    METHOD 1: Use OBB width/height DIRECTLY (RECOMMENDED for OBB models)
    This gives you the TRUE dimensions of the rotated object without whitespace.
    """
    x_center, y_center, width, height, angle = obb_box
    
    # Use width and height AS-IS from OBB
    # These are the actual dimensions along the object's principal axes
    width = int(width)
    height = int(height)
    
    # For bounding box (for display/filtering only)
    x_center, y_center = int(x_center), int(y_center)
    x1 = x_center - width // 2
    y1 = y_center - height // 2
    x2 = x_center + width // 2
    y2 = y_center + height // 2
    
    return width, height, (x1, y1, x2, y2)

def extract_obb_dimensions_method2(obb_box):
    """
    METHOD 2: Calculate from rotated rectangle corners (alternative approach)
    Gets min/max extents of the ROTATED rectangle's corners.
    """
    x_center, y_center, width, height, angle = obb_box
    
    # Create rotated rectangle
    rect = ((x_center, y_center), (width, height), np.degrees(angle))
    box_points = cv2.boxPoints(rect)
    
    # Calculate width and height from corner points
    # Width: max distance between points in x-direction along rotation
    # Height: max distance between points in y-direction along rotation
    
    # Get the two principal dimensions
    edge1 = np.linalg.norm(box_points[0] - box_points[1])
    edge2 = np.linalg.norm(box_points[1] - box_points[2])
    
    actual_width = int(max(edge1, edge2))
    actual_height = int(min(edge1, edge2))
    
    # Bounding box for display
    x_min = int(np.min(box_points[:, 0]))
    y_min = int(np.min(box_points[:, 1]))
    x_max = int(np.max(box_points[:, 0]))
    y_max = int(np.max(box_points[:, 1]))
    
    return actual_width, actual_height, (x_min, y_min, x_max, y_max)

def extract_obb_dimensions_method3(obb_box, frame):
    """
    METHOD 3: Contour-based area calculation (most accurate for irregular shapes)
    Uses actual pixel area inside the rotated rectangle.
    """
    x_center, y_center, width, height, angle = obb_box
    
    # Create rotated rectangle
    rect = ((x_center, y_center), (width, height), np.degrees(angle))
    box_points = cv2.boxPoints(rect)
    box_points = np.int32(box_points)
    
    # Create mask and calculate area
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [box_points], 255)
    area = cv2.countNonZero(mask)
    
    # Calculate equivalent width/height from area and aspect ratio
    aspect_ratio = width / height if height > 0 else 1
    calc_height = int(np.sqrt(area / aspect_ratio))
    calc_width = int(area / calc_height) if calc_height > 0 else int(width)
    
    # Bounding box for display
    x_min = int(np.min(box_points[:, 0]))
    y_min = int(np.min(box_points[:, 1]))
    x_max = int(np.max(box_points[:, 0]))
    y_max = int(np.max(box_points[:, 1]))
    
    return calc_width, calc_height, (x_min, y_min, x_max, y_max)

def extract_obb_dimensions(obb_box, frame=None):
    """
    Router function to select dimension extraction method.
    """
    if DIMENSION_METHOD == "obb_direct":
        return extract_obb_dimensions_method1(obb_box)
    elif DIMENSION_METHOD == "rotated_rect_minmax":
        return extract_obb_dimensions_method2(obb_box)
    elif DIMENSION_METHOD == "contour_area":
        if frame is None:
            print("⚠️ Warning: contour_area method needs frame, falling back to obb_direct")
            return extract_obb_dimensions_method1(obb_box)
        return extract_obb_dimensions_method3(obb_box, frame)
    else:
        return extract_obb_dimensions_method1(obb_box)
    
def normalize_dimensions_by_class(width, height, cls):
    """
    Normalize dimensions based on class to handle OBB rotation swapping.
    
    Class 0 (ROLLS): Keep as-is (width x height of roll)
    Class 1 (SIDE): Ensure width < height (stack is taller than wide)
    Class 2 (TOP): Always use SMALLER dimension as depth
    """
    if cls == 0:  # ROLLS 
        # ROLL side is typically TALLER than wide
        # So height should be > width
        if height > width:
            # Swapped! Fix it
            return height, width  # Return (new_width, new_height)
        return width, height
    if cls == 1:  # SIDE VIEW
        # Stack side is typically TALLER than wide
        # So height should be > width
        if width > height:
            # Swapped! Fix it
            return height, width  # Return (new_width, new_height)
        return width, height
    
    elif cls == 2:  # TOP VIEW
        # Depth is always the smaller dimension
        return min(width, height)
    
    else:  # Class 0 (ROLLS)
        return width, height

def find_closest_pattern(rolls_wide, rolls_high, depth_layers, med_roll_w, med_roll_h, med_side_w, med_side_h, med_top_h):
    """
    Find the closest known pattern and check if dimensions fit within ±25px buffer.
    """
    best_match = None
    best_score = float('inf')
    
    print(f"\n🔍 Pattern Matching Debug:")
    print(f"   Input dimensions: {rolls_wide}W × {rolls_high}H × {depth_layers}D")
    
    for pattern in KNOWN_PATTERNS:
        expected_side_w = pattern['wide'] * med_roll_w
        expected_side_h = pattern['high'] * med_roll_h
        expected_top_h = pattern['depth'] * med_roll_h
        
        width_diff = abs(med_side_w - expected_side_w)
        height_diff = abs(med_side_h - expected_side_h)
        depth_diff = abs(med_top_h - expected_top_h)
        
        total_diff = width_diff + height_diff + depth_diff
        
        within_buffer = (width_diff <= DIMENSION_TOLERANCE_PIXELS and 
                        height_diff <= DIMENSION_TOLERANCE_PIXELS and 
                        depth_diff <= DIMENSION_TOLERANCE_PIXELS)
        
        print(f"   Pattern {pattern['name']}: diffs W={width_diff:.1f}px H={height_diff:.1f}px D={depth_diff:.1f}px | {'✓ MATCH' if within_buffer else '✗'}")
        
        if within_buffer and total_diff < best_score:
            best_score = total_diff
            best_match = {
                'pattern': pattern,
                'width_diff': width_diff,
                'height_diff': height_diff,
                'depth_diff': depth_diff,
                'total_diff': total_diff
            }
    
    if best_match:
        pattern = best_match['pattern']
        match_info = (
            f"Matched pattern: {pattern['name']} | "
            f"Pixel diffs: W={best_match['width_diff']:.1f}px H={best_match['height_diff']:.1f}px D={best_match['depth_diff']:.1f}px"
        )
        print(f"   ✅ BEST MATCH: {pattern['name']} (total diff: {best_match['total_diff']:.1f}px)")
        return pattern, pattern['wide'], pattern['high'], pattern['depth'], match_info
    
    print(f"   ❌ NO PATTERN MATCH FOUND")
    return None, rolls_wide, rolls_high, depth_layers, ""

def validate_and_correct_dimensions(med_side_w, med_side_h, med_top_h, med_roll_w, med_roll_h, visible_rolls_count=0):
    """
    Validate stack dimensions against roll dimensions with learning.
    """
    if med_roll_w == 0 or med_roll_h == 0:
        return 0, 0, 0, False, "⚠️ No roll dimensions available"
    
    rolls_wide_float = med_side_w / med_roll_w
    rolls_high_float = med_side_h / med_roll_h
    depth_float = med_top_h / med_roll_h
    
    rolls_wide_detected = round(rolls_wide_float)
    rolls_high_detected = round(rolls_high_float)
    depth_detected = round(depth_float)
    
    print("\n" + "="*60)
    print("🔍 DIMENSION CALCULATION DEBUG")
    print("="*60)
    print(f"👁️ VISIBLE ROLLS IN FRAME: {visible_rolls_count}")
    print(f"Median Roll: {med_roll_w}px W × {med_roll_h}px H")
    print(f"Median Side: {med_side_w}px W × {med_side_h}px H")
    print(f"Median Top HEIGHT: {med_top_h}px")
    print(f"\nRaw Division Results:")
    print(f"  Wide  = {med_side_w} ÷ {med_roll_w} = {rolls_wide_float:.3f} → rounds to {rolls_wide_detected}")
    print(f"  High  = {med_side_h} ÷ {med_roll_h} = {rolls_high_float:.3f} → rounds to {rolls_high_detected}")
    print(f"  Depth = {med_top_h} ÷ {med_roll_h} = {depth_float:.3f} → rounds to {depth_detected}")
    print(f"\nDetected Configuration: {rolls_wide_detected}×{rolls_high_detected}×{depth_detected}")
    print(f"Calculated Total: {rolls_wide_detected * rolls_high_detected * depth_detected} rolls")
    
    matched_pattern, corrected_wide, corrected_high, corrected_depth, match_info = \
        find_closest_pattern(rolls_wide_detected, rolls_high_detected, depth_detected,
                           med_roll_w, med_roll_h, med_side_w, med_side_h, med_top_h)
    
    was_corrected = (matched_pattern is not None and 
                    (corrected_wide != rolls_wide_detected or 
                     corrected_high != rolls_high_detected or 
                     corrected_depth != depth_detected))
    
    correction_details = ""
    if was_corrected:
        correction_details = (
            f"🎯 PATTERN LEARNING APPLIED!\n"
            f"  Detected: {rolls_wide_detected}×{rolls_high_detected}×{depth_detected} "
            f"({rolls_wide_float:.2f}×{rolls_high_float:.2f}×{depth_float:.2f})\n"
            f"  Corrected to: {corrected_wide}×{corrected_high}×{corrected_depth} "
            f"(Pattern: {matched_pattern['name']})\n"
            f"  {match_info}"
        )
        print(f"\n🎯 CORRECTION APPLIED: {rolls_wide_detected}×{rolls_high_detected}×{depth_detected} → {corrected_wide}×{corrected_high}×{corrected_depth}")
    elif matched_pattern is not None:
        correction_details = f"✓ Pattern validated: {matched_pattern['name']} | {match_info}"
        print(f"\n✅ PATTERN VALIDATED: {matched_pattern['name']}")
    else:
        expected_side_w = rolls_wide_detected * med_roll_w
        expected_side_h = rolls_high_detected * med_roll_h
        expected_top_h = depth_detected * med_roll_h
        
        width_diff_px = abs(med_side_w - expected_side_w)
        height_diff_px = abs(med_side_h - expected_side_h)
        depth_diff_px = abs(med_top_h - expected_top_h)
        
        within_buffer = (width_diff_px <= DIMENSION_TOLERANCE_PIXELS and 
                        height_diff_px <= DIMENSION_TOLERANCE_PIXELS and 
                        depth_diff_px <= DIMENSION_TOLERANCE_PIXELS)
        
        if not within_buffer:
            correction_details = (
                f"⚠️ No pattern match found!\n"
                f"  Detected: {rolls_wide_detected}×{rolls_high_detected}×{depth_detected}\n"
                f"  Pixel diffs: W={width_diff_px:.1f}px H={height_diff_px:.1f}px D={depth_diff_px:.1f}px\n"
                f"  (Exceeds ±{DIMENSION_TOLERANCE_PIXELS}px buffer)"
            )
            print(f"\n⚠️ NO PATTERN MATCH - Exceeds tolerance")
        else:
            correction_details = (
                f"✓ Custom pattern (within buffer): {rolls_wide_detected}×{rolls_high_detected}×{depth_detected}\n"
                f"  Pixel diffs: W={width_diff_px:.1f}px H={height_diff_px:.1f}px D={depth_diff_px:.1f}px"
            )
            print(f"\n✓ CUSTOM PATTERN ACCEPTED (within buffer)")
        
        corrected_wide = rolls_wide_detected
        corrected_high = rolls_high_detected
        corrected_depth = depth_detected
    
    print("="*60 + "\n")
    
    return corrected_wide, corrected_high, corrected_depth, was_corrected, correction_details

def reset_stack_data():
    """Reset ALL stack-specific data when stack disappears"""
    global roll_widths, roll_heights
    global side_widths, side_heights, top_heights
    global median_roll_width, median_roll_height, median_roll_area
    global median_side_width, median_side_height, median_side_area, median_top_height
    global stack_calculation_done, stack_first_seen_frame
    
    roll_widths.clear()
    roll_heights.clear()
    median_roll_width = 0
    median_roll_height = 0
    median_roll_area = 0
    
    side_widths.clear()
    side_heights.clear()
    top_heights.clear()
    median_side_width = 0
    median_side_height = 0
    median_side_area = 0
    median_top_height = 0
    stack_calculation_done = False
    stack_first_seen_frame = 0
    
    print("🔄 Stack data RESET (including roll dimensions)")

def count_rolls_with_unified_model(video_source=0):
    global roll_widths, roll_heights
    global median_roll_width, median_roll_height, median_roll_area
    global side_widths, side_heights, top_heights
    global median_side_width, median_side_height, median_side_area, median_top_height
    global stack_was_visible, stack_disappeared_frames, stack_calculation_done, stack_first_seen_frame
    global calculated_rolls_current_stack, total_material_in_rolls
    global total_frames_processed, total_side_detections, total_top_detections, total_roll_detections
    global total_corrections_applied, total_pattern_matches
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("="*60)
    print("🚀 OBB ROLL COUNTING SYSTEM - DIMENSION FIX")
    print("="*60)
    print(f"✅ DIMENSION METHOD: {DIMENSION_METHOD}")
    print("   obb_direct = Use OBB width/height directly (RECOMMENDED)")
    print("   rotated_rect_minmax = Calculate from corner points")
    print("   contour_area = Use pixel area inside rotated rect")
    print(f"✅ PATTERN LEARNING: {len(KNOWN_PATTERNS)} patterns loaded")
    print(f"✅ TOLERANCE: ±{DIMENSION_TOLERANCE_PIXELS}px buffer")
    print("="*60)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n🔹 Video stream ended.")
            break
        
        frame_count += 1
        total_frames_processed += 1
        
        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        
        rolls_in_frame = []
        side_view_detected = False
        top_view_detected = False
        
        if len(results) > 0:
            use_obb = hasattr(results[0], 'obb') and results[0].obb is not None
            
            if use_obb:
                obb_xywhr = results[0].obb.xywhr.cpu().numpy()
                classes = results[0].obb.cls.cpu().numpy().astype(int)
                confidences = results[0].obb.conf.cpu().numpy()
            elif hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
            else:
                boxes = None
                classes = []
                confidences = []
            
            for idx, (cls, conf) in enumerate(zip(classes, confidences)):
                if use_obb:
                    obb_data = obb_xywhr[idx]
                    # FIXED: Use proper dimension extraction
                    width, height, bbox = extract_obb_dimensions(obb_data, frame)
                    x1, y1, x2, y2 = bbox
                else:
                    box = boxes[idx]
                    x1, y1, x2, y2 = map(int, box)
                    width = x2 - x1
                    height = y2 - y1
                
                if cls == 0:  # ROLLS
                    if conf < CONFIDENCE_ROLLS:
                        continue
                    
                    roll_width, roll_height = normalize_dimensions_by_class(width, height, cls)
                    
                    if roll_width < MIN_ROLL_WIDTH or roll_height < MIN_ROLL_HEIGHT:
                        continue
                    if MAX_ROLL_WIDTH > 0 and roll_width > MAX_ROLL_WIDTH:
                        continue
                    if MAX_ROLL_HEIGHT > 0 and roll_height > MAX_ROLL_HEIGHT:
                        continue
                    
                    roll_widths.append(roll_width)
                    roll_heights.append(roll_height)
                    total_roll_detections += 1
                    
                    rolls_in_frame.append({
                        'box': (x1, y1, x2, y2),
                        'width': roll_width,
                        'height': roll_height,
                    })
                    
                    if display:
                        if use_obb:
                            # Draw TRUE rotated rectangle
                            obb_data = obb_xywhr[idx]
                            x_center, y_center, obb_w, obb_h, angle = obb_data
                            rect = ((x_center, y_center), (obb_w, obb_h), np.degrees(angle))
                            box_points = cv2.boxPoints(rect)
                            box_points = np.int32(box_points)
                            cv2.drawContours(frame, [box_points], 0, (255, 0, 0), 2)
                            
                            cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
                            
                            roll_label = f"{roll_width}x{roll_height}"
                            cv2.putText(frame, roll_label, (int(x_center) - 20, int(y_center) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                            roll_label = f"{roll_width}x{roll_height}"
                            cv2.putText(frame, roll_label, (x1 + 2, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                elif cls == 1:  # SIDE VIEW
                    if conf < CONFIDENCE_SIDE:
                        continue
                    
                    side_width, side_height = normalize_dimensions_by_class(width, height, cls)
                    side_area = side_width * side_height
                    
                    if MIN_SIDE_AREA > 0 and side_area < MIN_SIDE_AREA:
                        continue
                    
                    side_view_detected = True
                    total_side_detections += 1
                    
                    side_widths.append(side_width)
                    side_heights.append(side_height)
                    
                    if display:
                        if use_obb:
                            obb_data = obb_xywhr[idx]
                            x_center, y_center, obb_w, obb_h, angle = obb_data
                            rect = ((x_center, y_center), (obb_w, obb_h), np.degrees(angle))
                            box_points = cv2.boxPoints(rect)
                            box_points = np.int32(box_points)
                            cv2.drawContours(frame, [box_points], 0, (0, 255, 0), 3)
                            
                            side_label = f"Side {side_width}x{side_height}"
                            cv2.putText(frame, side_label, (int(x_center) - 30, int(y_center) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            side_label = f"Side {side_width}x{side_height}"
                            cv2.putText(frame, side_label, (x1 + 5, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                elif cls == 2:  # TOP VIEW
                    if conf < CONFIDENCE_TOP:
                        continue
                    
                    top_height = normalize_dimensions_by_class(width, height, cls)  # <--- FIXED LINE
                    
                    if MIN_TOP_DEPTH > 0 and top_height < MIN_TOP_DEPTH:
                        continue
                    
                    top_view_detected = True
                    total_top_detections += 1
                    
                    top_heights.append(top_height)
                    
                    if display:
                        if use_obb:
                            obb_data = obb_xywhr[idx]
                            x_center, y_center, obb_w, obb_h, angle = obb_data
                            rect = ((x_center, y_center), (obb_w, obb_h), np.degrees(angle))
                            box_points = cv2.boxPoints(rect)
                            box_points = np.int32(box_points)
                            cv2.drawContours(frame, [box_points], 0, (0, 255, 255), 3)
                            
                            display_depth = normalize_dimensions_by_class(width, height, cls)
                            top_label = f"Top D:{display_depth}"
                            cv2.putText(frame, top_label, (int(x_center) - 20, int(y_center) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                            top_label = f"Top H:{top_height}"
                            cv2.putText(frame, top_label, (x1 + 5, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        median_roll_width = calculate_median_safe(roll_widths)
        median_roll_height = calculate_median_safe(roll_heights)
        median_side_width = calculate_median_safe(side_widths)
        median_side_height = calculate_median_safe(side_heights)
        median_top_height = calculate_median_safe(top_heights)
        
        stack_detected = side_view_detected or top_view_detected
        
        if stack_detected and stack_first_seen_frame == 0:
            stack_first_seen_frame = frame_count
            print(f"\n🆕 NEW STACK DETECTED at frame {frame_count}")
        
        if stack_detected and not stack_calculation_done:
            if frame_count % 10 == 0:
                print(f"📊 Frame {frame_count}: Collecting data... Roll:{len(roll_widths)} Side:{len(side_widths)} Top:{len(top_heights)} | Visible rolls: {len(rolls_in_frame)}")

        elif not stack_detected and stack_was_visible and stack_disappeared_frames == 1 and not stack_calculation_done:
            min_samples_needed = min(10, ROLLING_WINDOW_SIZE // 3)
            
            print(f"\n🎯 Stack disappearing - Running calculation NOW")
            print(f"   Data collected: Roll:{len(roll_widths)} Side:{len(side_widths)} Top:{len(top_heights)}")
            
            if (len(side_widths) >= min_samples_needed and 
                len(top_heights) >= min_samples_needed and
                len(roll_widths) >= min_samples_needed and
                median_roll_width > 0 and median_roll_height > 0):
                
                corrected_wide, corrected_high, corrected_depth, was_corrected, correction_info = \
                    validate_and_correct_dimensions(median_side_width, median_side_height, 
                                                   median_top_height, median_roll_width, median_roll_height, len(rolls_in_frame))
                
                if was_corrected:
                    total_corrections_applied += 1
                    total_pattern_matches += 1
                
                calculated_rolls = corrected_wide * corrected_high * corrected_depth
                
                if validate_calculation(calculated_rolls):
                    calculated_rolls_current_stack = calculated_rolls
                    stack_calculation_done = True
                    
                    print(f"\n✨ Frame {frame_count}: STACK CALCULATED")
                    print(f"  Stack visible since frame {stack_first_seen_frame} ({frame_count - stack_first_seen_frame} frames)")
                    print(f"  {correction_info}")
                    print(f"  ➡️ TOTAL: {calculated_rolls} rolls ({corrected_wide}×{corrected_high}×{corrected_depth})")
        
        if stack_detected:
            stack_was_visible = True
            stack_disappeared_frames = 0
        else:
            if stack_was_visible:
                stack_disappeared_frames += 1
                
                if stack_disappeared_frames == DISAPPEAR_THRESHOLD:
                    if calculated_rolls_current_stack > 0:
                        total_material_in_rolls += calculated_rolls_current_stack
                        
                        print(f"\n{'='*60}")
                        print(f"🎯 MATERIAL IN EVENT!")
                        print(f"{'='*60}")
                        print(f"   Stack Rolls: {calculated_rolls_current_stack}")
                        print(f"   Cumulative Total: {total_material_in_rolls}")
                        print(f"{'='*60}\n")
                    else:
                        print(f"\n⚠️ Stack disappeared but no calculation available!")
                    
                    calculated_rolls_current_stack = 0
                    reset_stack_data()
                    stack_was_visible = False
        
        if display:
            y = 40
            cv2.putText(frame, f"Frame: {frame_count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 35
            
            if stack_detected:
                data_status = f"[R:{len(roll_widths)} S:{len(side_widths)} T:{len(top_heights)}]"
                stack_color = (0, 255, 0) if stack_calculation_done else (0, 165, 255)
                cv2.putText(frame, f"Stack: VISIBLE {data_status}", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, stack_color, 2)
            else:
                cv2.putText(frame, f"Stack: HIDDEN ({stack_disappeared_frames}/{DISAPPEAR_THRESHOLD})", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            y += 35
            
            cv2.putText(frame, f"Roll Med: {median_roll_width}x{median_roll_height}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y += 30
            
            calc_text = f"Calculated: {calculated_rolls_current_stack}"
            if stack_calculation_done:
                calc_text += " ✓"
            cv2.putText(frame, calc_text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            y += 40
            
            cv2.putText(frame, f"Material IN: {total_material_in_rolls}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            y += 40
            
            cv2.putText(frame, f"Visible rolls: {len(rolls_in_frame)}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 30

            cv2.putText(frame, f"Side: {median_side_width}x{median_side_height}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y += 25
            cv2.putText(frame, f"Top H: {median_top_height}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y += 30
            
            # NEW: Show dimension method
            cv2.putText(frame, f"Method: {DIMENSION_METHOD}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("OBB Roll Counter - FIXED", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("📊 FINAL STATISTICS")
    print("="*60)
    print(f"Dimension Method Used: {DIMENSION_METHOD}")
    print(f"Total frames: {total_frames_processed}")
    print(f"Total roll detections: {total_roll_detections}")
    print(f"Pattern matches: {total_pattern_matches}")
    print(f"🎯 TOTAL MATERIAL IN: {total_material_in_rolls} rolls")
    print("="*60)

if __name__ == "__main__":
    count_rolls_with_unified_model("Screen Recording 2025-10-29 163942.mp4")


    