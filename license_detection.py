# license_detection.py
# License Plate Detection Module - VS Code Version

import cv2
import numpy as np
from PIL import Image
import easyocr
import pandas as pd
import uuid
import os
import re
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import csv

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

# ============== CONFIGURATION ==============
MODEL_PATH = "liscence3.pt"  # Update this path
CONFIDENCE_THRESHOLD = 0.15
SAVE_CROPPED_IMAGES = False

# Folder paths
FOLDER_PATH = "./licenses_plates_imgs_detected/"
CSV_FOLDER = "./csv_detections/"

# Create folders
if SAVE_CROPPED_IMAGES:
    os.makedirs(FOLDER_PATH, exist_ok=True)
os.makedirs(CSV_FOLDER, exist_ok=True)

# Global variables for model and reader
model = None
reader = None

def initialize_models():
    """Initialize YOLO model and EasyOCR reader"""
    global model, reader
    
    if model is None:
        print(f"Loading license plate model from: {MODEL_PATH}...")
        try:
            model = YOLO(MODEL_PATH)
            print("✅ License plate model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please ensure the model file exists and path is correct.")
            raise
    
    if reader is None:
        print("Loading EasyOCR (English only)...")
        reader = easyocr.Reader(['en'], gpu=True)
        print("✅ EasyOCR loaded successfully!")

# ============== CHARACTER MAPPING ==============
def correct_character_by_position(char, position, total_length, plate_format):
    """Correct OCR mistakes based on position"""
    digit_to_letter = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G'}
    letter_to_digit = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '6', 'Z': '2', 'A': '4'}

    if plate_format == 'indian':
        if total_length >= 8:
            if position < 2 and char.isdigit():
                return digit_to_letter.get(char, char)
            elif 2 <= position < 4 and char.isalpha():
                return letter_to_digit.get(char, char)
            elif 4 <= position < 6 and char.isdigit():
                return digit_to_letter.get(char, char)
            elif position >= 6 and char.isalpha():
                return letter_to_digit.get(char, char)

    elif plate_format == 'uk':
        if total_length >= 7:
            if position < 2 and char.isdigit():
                return digit_to_letter.get(char, char)
            elif 2 <= position < 4 and char.isalpha():
                return letter_to_digit.get(char, char)
            elif position >= 4 and char.isdigit():
                return digit_to_letter.get(char, char)

    return char

# ============== FORMAT PATTERNS ==============
INDIAN_PATTERN = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$')
UK_PATTERN = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{3}$')

def detect_plate_format(text):
    """Detect which format the plate matches"""
    text = text.upper().replace(' ', '')
    if INDIAN_PATTERN.match(text):
        return 'indian'
    elif UK_PATTERN.match(text):
        return 'uk'
    return 'unknown'

def validate_license_format(text):
    """Validate if text matches known license plate formats"""
    text = text.upper().replace(' ', '')

    if INDIAN_PATTERN.match(text):
        return True, 'indian'
    elif UK_PATTERN.match(text):
        return True, 'uk'

    if len(text) >= 7:
        has_letters = sum(c.isalpha() for c in text) >= 3
        has_digits = sum(c.isdigit() for c in text) >= 3
        if has_letters and has_digits:
            return True, 'mixed'

    return False, 'invalid'

# ============== UTILITY FUNCTIONS ==============
def write_csv(results, output_path):
    """Write detection results to CSV - only indian/uk formats"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([writer.writerow(['timestamp', 'frame_num', 'license_plate', 'confidence', 'format_type', 'material_count'])])

        added_count = 0
        skipped_count = 0

        for plate_id, data_dict in results.items():
            if plate_id in data_dict:
                detection_data = data_dict[plate_id]
                car_data = detection_data.get('car', {})
                plate_data = detection_data.get('license_plate', {})
                format_type = plate_data.get('format_type', 'unknown')

                # if format_type not in ['indian', 'uk']:
                #     skipped_count += 1
                #     print(f"  ⏭️  Skipping plate_id {plate_id}: format '{format_type}'")
                #     continue

                regular_bbox = plate_data.get('bbox', [0, 0, 0, 0])
                license_text = plate_data.get('text', '') if plate_data.get('text') is not None else ''
                text_score = plate_data.get('text_score', 0) if plate_data.get('text_score') is not None else 0
                frame_num = plate_data.get('frame', 0)
                timestamp = f"{frame_num / 30:.2f}s"
                writer.writerow([
                    timestamp,
                    frame_num,
                    license_text,
                    plate_data.get('bbox_score', 0),  # This is confidence
                    format_type,
                    0  # material_count placeholder (will be filled in unified_detection.py)
                ])
                added_count += 1

    print(f"✅ CSV saved: {output_path}")
    print(f"   📊 Added: {added_count} | Skipped: {skipped_count}")

def preprocess_license_plate(license_plate_crop):
    """Apply preprocessing techniques"""
    preprocessed_images = []
    gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

    # CLAHE + Otsu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, enhanced_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(("CLAHE+Otsu", enhanced_thresh))

    # Bilateral filter + Otsu
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    _, bilateral_thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(("Bilateral+Otsu", bilateral_thresh))

    # Adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    preprocessed_images.append(("Adaptive", adaptive_thresh))

    return preprocessed_images

def read_license_plate(license_plate_crop):
    """Read license plate text with OCR"""
    if license_plate_crop is None or license_plate_crop.size == 0:
        return None, None, 'invalid'

    try:
        preprocessed_versions = preprocess_license_plate(license_plate_crop)
        all_candidates = []

        for name, processed_img in preprocessed_versions:
            try:
                detections = reader.readtext(processed_img, detail=1)

                if detections:
                    texts = []
                    scores = []

                    for bbox, text, score in detections:
                        text = text.upper().strip()
                        text = text.replace('|', '').replace('_', '').replace('.', '').replace(' ', '')

                        if text and len(text) >= 4:
                            texts.append(text)
                            scores.append(score)

                    if texts:
                        combined = "".join(texts)
                        avg_score = sum(scores) / len(scores)
                        all_candidates.append((combined, avg_score, name))
                        print(f"  [{name}] Detected: {combined} (score: {avg_score:.3f})")

            except Exception as e:
                continue

        if not all_candidates:
            return None, None, 'invalid'

        all_candidates.sort(key=lambda x: x[1], reverse=True)
        best_text, best_score, best_method = all_candidates[0]

        filtered_text = ''.join(c for c in best_text if c.isalnum())

        if not filtered_text or len(filtered_text) < 4:
            return None, None, 'invalid'

        plate_format = detect_plate_format(filtered_text)

        if plate_format in ['indian', 'uk']:
            corrected_text = ''
            for i, char in enumerate(filtered_text):
                corrected_char = correct_character_by_position(
                    char, i, len(filtered_text), plate_format
                )
                corrected_text += corrected_char

            is_valid, final_format = validate_license_format(corrected_text)
            print(f"  ✓ Best Result: {corrected_text} [{final_format}] (score: {best_score:.3f})")
            return corrected_text, best_score, final_format
        else:
            print(f"  ⚠ Unknown format: {filtered_text}")
            return filtered_text, best_score, 'unknown'

    except Exception as e:
        print(f"  ❌ Error in OCR: {e}")
        return None, None, 'invalid'

def draw_detection_box(image, bbox, text=None, color=(0, 255, 0), thickness=3):
    """Draw bounding box with text overlay"""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

            text_x = x1
            text_y = y1 - 10

            if text_y - text_height - 5 < 0:
                text_y = y1 + text_height + 20

            bg_x1 = text_x
            bg_y1 = text_y - text_height - 5
            bg_x2 = text_x + text_width + 10
            bg_y2 = text_y + 5

            cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
            cv2.putText(image, text, (text_x + 5, text_y), font, font_scale, (0, 0, 0), font_thickness)

    except Exception as e:
        print(f"  ❌ Error drawing detection: {e}")

# ============== MAIN DETECTION FUNCTION ==============
def detect_license_plates(image_path, frame_number=0):
    """Detect license plates and extract text"""
    initialize_models()
    
    print(f"\n{'='*60}")
    print(f"Processing: {image_path} (Frame: {frame_number})")
    print(f"{'='*60}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image {image_path}")
        return None, None, None

    img_display = img.copy()

    print("Running YOLO inference...")
    results = model(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)

    predictions = results[0].boxes
    print(f"Model Response: {len(predictions)} predictions")

    results_dict = {}
    license_crops = []
    license_texts = []
    detection_count = 0

    if len(predictions) > 0:
        print(f"\n📍 Found {len(predictions)} detection(s)")

        for i, box in enumerate(predictions):
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            print(f"\n{'─'*60}")
            print(f"Detection {i+1}:")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Box: [{x1}, {y1}, {x2}, {y2}]")

            license_crop = img[y1:y2, x1:x2]

            if license_crop.size > 0:
                if SAVE_CROPPED_IMAGES:
                    crop_name = f'{uuid.uuid4()}.jpg'
                    crop_path = os.path.join(FOLDER_PATH, crop_name)
                    cv2.imwrite(crop_path, license_crop)
                    print(f"  ✅ Saved crop: {crop_name}")

                print(f"  🔍 Running OCR...")
                plate_text, text_score, format_type = read_license_plate(license_crop)

                draw_detection_box(img_display, [x1, y1, x2, y2], text=plate_text)

                license_crops.append(license_crop)
                license_texts.append(plate_text)

                results_dict[detection_count] = {
                    detection_count: {
                        'car': {'bbox': [0, 0, 0, 0], 'car_score': 0},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': plate_text or '',
                            'bbox_score': confidence,
                            'text_score': text_score or 0,
                            'frame': frame_number,
                            'format_type': format_type
                        }
                    }
                }
                detection_count += 1
            else:
                print("  ❌ Invalid crop size")

    if results_dict:
        csv_path = f"{CSV_FOLDER}/detection_results.csv"
        write_csv(results_dict, csv_path)
        print(f"\n✅ Results saved to CSV: {csv_path}")

    print(f"\n{'='*60}")
    print(f"📊 SUMMARY: {detection_count} plate(s) detected")
    print(f"{'='*60}\n")

    return img_display, license_texts, license_crops

def process_video(video_path, output_path="output_video.mp4", display_video=True):
    """Process video and detect license plates frame by frame"""
    initialize_models()
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_detections = {}
    all_results = {}
    frame_number = 0
    global_detection_id = 0

    print(f"\n{'='*60}")
    print(f"🎬 VIDEO PROCESSING")
    print(f"{'='*60}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"{'='*60}\n")

    with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                temp_frame_path = f"temp_frame_{frame_number}.jpg"
                cv2.imwrite(temp_frame_path, frame)

                processed_frame, texts, crops = detect_license_plates(temp_frame_path, frame_number)

                if texts and any(t is not None for t in texts):
                    all_detections[frame_number] = [t for t in texts if t is not None]

                    for text in texts:
                        if text:
                            all_results[global_detection_id] = {
                                global_detection_id: {
                                    'car': {'bbox': [0, 0, 0, 0], 'car_score': 0},
                                    'license_plate': {
                                        'frame': frame_number,
                                        'text': text,
                                        'bbox': [0, 0, 0, 0],
                                        'bbox_score': 0,
                                        'text_score': 0,
                                        'format_type': detect_plate_format(text)
                                    }
                                }
                            }
                            global_detection_id += 1

                if processed_frame is not None:
                    out.write(processed_frame)
                else:
                    out.write(frame)

                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)

            except Exception as e:
                print(f"❌ Error frame {frame_number}: {e}")
                out.write(frame)

            frame_number += 1
            pbar.update(1)

    cap.release()
    out.release()

    if all_results:
        video_csv_path = f"{CSV_FOLDER}/video_detection_results.csv"
        write_csv(all_results, video_csv_path)
        print(f"\n✅ Video results saved: {video_csv_path}")

    print(f"\n{'='*60}")
    print(f"✅ VIDEO COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Total detections: {global_detection_id}")
    print(f"{'='*60}\n")

    if display_video:
        cap_display = cv2.VideoCapture(output_path)
        while cap_display.isOpened():
            ret, frame = cap_display.read()
            if not ret:
                break
            cv2.imshow('License Detection Result', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap_display.release()
        cv2.destroyAllWindows()

    return output_path, all_detections

def display_results(img, texts, crops):
    """Display detection results"""
    cv2.imshow('License Plate Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if crops:
        print(f"\n{'='*60}")
        print(f"🔍 DETECTED LICENSE PLATES ({len(crops)})")
        print(f"{'='*60}\n")

        for i, (text, crop) in enumerate(zip(texts, crops)):
            if text:
                is_valid, format_type = validate_license_format(text)
                status = "✅" if is_valid else "⚠️"
                print(f"  {status} Plate {i+1}: {text} [{format_type}]")
                cv2.imshow(f'Plate {i+1}: {text}', crop)
                cv2.waitKey(0)
            else:
                print(f"  ❌ Plate {i+1}: No text detected")
                cv2.imshow(f'Plate {i+1}: No text', crop)
                cv2.waitKey(0)

        cv2.destroyAllWindows()
        print(f"\n{'='*60}\n")

    try:
        csv_path = f"{CSV_FOLDER}/detection_results.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print("\n" + "="*60)
            print("📊 DETECTION DATA")
            print("="*60)
            print(df.to_string())
            print(f"\n💾 CSV: {csv_path}")
    except Exception as e:
        print(f"⚠️  Could not load CSV: {e}")