import cv2
import os
import sys

def extract_key_frames(video_path, output_folder):
    if not os.path.exists(video_path):
        print(f"❌ Video does not exist: {video_path}")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Use "sourceX" as video ID
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 6:
        print(f"❗ {video_id} is too short with ({total_frames} frames), skipping.")
        return

    # Pick 6 key frames: 2 at start, 2 in middle, 2 at end
    frames_to_save = [
        0, 1,
        total_frames // 2 - 1, total_frames // 2,
        total_frames - 2, total_frames - 1
    ]

    saved_count = 0
    for idx, frame_index in enumerate(frames_to_save):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            filename = f"{video_id}_frame_{idx+1}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), frame)
            saved_count += 1
        else:
            print(f"⚠️ Could not read frame at index {frame_index} in {video_id}")

    cap.release()
    print(f"✅ Saved {saved_count} frames from {video_id}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❌ Usage: python extract_frames.py <video_file> <output_folder>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_folder = sys.argv[2]
    extract_key_frames(video_path, output_folder)


#Problem: Frames extracted are not very diverse
