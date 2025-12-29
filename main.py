"""
Main Orchestrator for License Detection and Material Counting
=============================================================
This script provides a unified interface to run both:
1. License Plate Detection (Images/Videos)
2. Material Roll Counting (Video streams)
"""

import os
import sys

def print_header():
    """Print application header"""
    print("\n" + "="*70)
    print("🚀 UNIFIED DETECTION SYSTEM")
    print("="*70)
    print("   [1] License Plate Detection (Image/Video)")
    print("   [2] Material Roll Counting (Video)")
    print("   [3] Run Both Systems Sequentially")
    print("   [4] Exit")
    print("="*70)

def print_license_submenu():
    """Print license detection submenu"""
    print("\n" + "="*70)
    print("🚗 LICENSE PLATE DETECTION")
    print("="*70)
    print("   [1] Process Single Image")
    print("   [2] Process Video")
    print("   [3] Back to Main Menu")
    print("="*70)

def run_license_detection_image():
    """Run license plate detection on image"""
    print("\n📸 IMAGE PROCESSING MODE")
    print("-" * 70)
    
    image_path = input("Enter image path (or press Enter for default): ").strip()
    if not image_path:
        print("⚠️  No image path provided. Please provide a valid image path.")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        return
    
    try:
        import license_detection as ld
        
        print(f"\n🔍 Processing image: {image_path}")
        img, texts, crops = ld.detect_license_plates(image_path)
        
        if img is not None:
            ld.display_results(img, texts, crops)
            print("\n✅ Image processing complete!")
        else:
            print("❌ Failed to process image")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def run_license_detection_video():
    """Run license plate detection on video"""
    print("\n🎬 VIDEO PROCESSING MODE")
    print("-" * 70)
    
    video_path = input("Enter video path (or press Enter for default): ").strip()
    if not video_path:
        print("⚠️  No video path provided. Please provide a valid video path.")
        return
    
    if not os.path.exists(video_path):
        print(f"❌ Error: File not found: {video_path}")
        return
    
    output_path = input("Enter output video path (default: output_license.mp4): ").strip()
    if not output_path:
        output_path = "output_license.mp4"
    
    try:
        import license_detection as ld
        
        print(f"\n🔍 Processing video: {video_path}")
        output, detections = ld.process_video(video_path, output_path, display_video=True)
        
        print("\n✅ Video processing complete!")
        print(f"📁 Output saved to: {output}")
        
        # Summary
        if detections:
            total_detections = sum(len(plates) for plates in detections.values())
            print(f"\n📊 SUMMARY:")
            print(f"   Frames with detections: {len(detections)}")
            print(f"   Total plates detected: {total_detections}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def run_material_counting():
    """Run material roll counting"""
    print("\n📦 MATERIAL ROLL COUNTING")
    print("-" * 70)
    
    video_source = input("Enter video path or camera index (0 for webcam, default: 0): ").strip()
    
    # Convert to int if it's a number (camera index), otherwise keep as string (file path)
    if video_source.isdigit():
        video_source = int(video_source)
    elif not video_source:
        video_source = 0
    elif not os.path.exists(video_source):
        print(f"❌ Error: File not found: {video_source}")
        return
    
    try:
        import material_counting as mc
        
        print(f"\n🔍 Starting material counting...")
        print("Press 'q' to stop counting")
        
        mc.count_rolls_with_unified_model(video_source)
        
        print("\n✅ Material counting complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def run_both_systems():
    """Run both systems sequentially"""
    print("\n🔄 SEQUENTIAL EXECUTION MODE")
    print("-" * 70)
    print("This will run License Detection first, then Material Counting")
    
    # License Detection
    print("\n" + "="*70)
    print("STEP 1: LICENSE PLATE DETECTION")
    print("="*70)
    
    license_mode = input("Process [1] Image or [2] Video? (1/2): ").strip()
    
    if license_mode == "1":
        run_license_detection_image()
    elif license_mode == "2":
        run_license_detection_video()
    else:
        print("❌ Invalid choice")
        return
    
    # Material Counting
    print("\n" + "="*70)
    print("STEP 2: MATERIAL ROLL COUNTING")
    print("="*70)
    
    proceed = input("\nProceed to Material Counting? (y/n): ").strip().lower()
    if proceed == 'y':
        run_material_counting()
    else:
        print("⏭️  Skipped material counting")
    
    print("\n✅ Both systems executed!")

def license_detection_menu():
    """Handle license detection submenu"""
    while True:
        print_license_submenu()
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            run_license_detection_image()
        elif choice == "2":
            run_license_detection_video()
        elif choice == "3":
            break
        else:
            print("❌ Invalid choice. Please select 1-3.")
        
        input("\nPress Enter to continue...")

def main():
    """Main application loop"""
    while True:
        print_header()
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            license_detection_menu()
        elif choice == "2":
            run_material_counting()
            input("\nPress Enter to continue...")
        elif choice == "3":
            run_both_systems()
            input("\nPress Enter to continue...")
        elif choice == "4":
            print("\n👋 Exiting... Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Please select 1-4.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    # Check if required modules exist
    if not os.path.exists("license_detection.py"):
        print("❌ Error: license_detection.py not found!")
        sys.exit(1)
    
    if not os.path.exists("material_counting.py"):
        print("❌ Error: material_counting.py not found!")
        print("💡 Tip: Rename your roll counting code to 'material_counting.py'")
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)