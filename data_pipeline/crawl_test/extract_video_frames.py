import cv2
import os
import sys

def extract_key_frames(video_path, output_folder, num_frames=6):
    """
    Extracts a fixed number of evenly spaced key frames from a video to ensure visual diversity.

    Parameters:
        video_path (str): Path to the input video file.
        output_folder (str): Directory to save extracted frames.
        num_frames (int): Number of key frames to extract (default is 6).

    Returns:
        None. Saves extracted frames as JPEG files in the output folder.
    """

    if not os.path.exists(video_path):
        print(f"❌ Video does not exist: {video_path}")
        return
    os.makedirs(output_folder, exist_ok=True)
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        print(f"❗ {video_id} is too short with ({total_frames} frames), skipping.")
        return

    interval = total_frames // (num_frames + 1)
    frames_to_save = [interval * (i + 1) for i in range(num_frames)]

    saved_count = 0

    # Extract and save each selected frame
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
    print(f"✅ Saved {saved_count} diverse frames from {video_id}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❌ Usage: python extract_frames.py <video_file> <output_folder>")
        sys.exit(1)
    video_path = sys.argv[1]
    output_folder = sys.argv[2]
    extract_key_frames(video_path, output_folder)


#Maybe some errors with AV1, run better on GPU 