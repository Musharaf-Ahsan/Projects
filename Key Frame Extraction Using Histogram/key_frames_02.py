import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor

def detect_shot_boundaries_and_keyframes(video_path, alpha=1.10):
    cap = cv2.VideoCapture(video_path)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window_size = int(frame_rate)
    hist = np.zeros(frame_count)

    # Calculate HSV histograms for each frame
    for i in range(1, frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i - 1)
        _, frame1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame2 = cap.read()

        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

        hist1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])

        hist[i] = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

    window = np.pad(hist, (window_size // 2, window_size // 2), mode='constant')

    shot_boundaries = []
    key_frames = []

    # Create the output directory for keyframes
    output_directory = os.path.splitext(video_path)[0] + "_keyframes"
    os.makedirs(output_directory, exist_ok=True)

    # Open the text file to save shot boundary and key frame information
    output_file_path = os.path.splitext(video_path)[0] + "_shot_boundary_and_keyframes_info.txt"
    output_file = open(output_file_path, 'w')

    # Measure the start time
    start_time = time.time()

    # Detect shot boundaries and key frames
    for i in range(window_size, len(hist) + 1):
        window = hist[i - window_size:i]
        mid = window[window_size // 2 - 1]
        window = np.sort(window)

        m1 = window[0]
        m2 = window[1]
        mean_val = np.mean(window)
        std_val = np.std(window)

        if mid == m1 and mid <= alpha * m2 and mean_val >= alpha * m1:
            shot_boundaries.append(i)
            key_frame_index = np.argmax(window)
            key_frame = key_frame_index + i - window_size
            key_frames.append(key_frame)

            # Extract and save the key frame image
            cap.set(cv2.CAP_PROP_POS_FRAMES, key_frame)
            _, frame = cap.read()

            # Check if the frame is valid before saving it as an image
            if frame is not None:
                image_path = os.path.join(output_directory, f"keyframe_{key_frame}.jpg")
                print("Saving image:", image_path)
                cv2.imwrite(image_path, frame)

                # Write shot boundary and key frame info into the text file
                time_for_key_frame = (key_frame / frame_rate)
                output_file.write(f"Shot boundary: {i}, Key frame: {key_frame}, Time: {time_for_key_frame:.2f} seconds\n")

    # Measure the end time
    end_time = time.time()
    # Calculate the total processing time
    processing_time = end_time - start_time

    # Write the total processing time into the text file
    output_file.write(f"Total processing time: {processing_time} seconds\n")

    # Close the text file
    output_file.close()

    cap.release()
    cv2.destroyAllWindows()

    return shot_boundaries, key_frames, frame_rate

def process_video(video_path):
    detect_shot_boundaries_and_keyframes(video_path)

if __name__ == "__main__":
    # Specify the input directory containing MP4 files
    input_directory = "D:/keyfram/yes_no"

    # Create a ThreadPoolExecutor with a maximum of 2 workers
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Iterate over each MP4 file in the input directory
        for filename in os.listdir(input_directory):
            if filename.endswith(".mp4"):
                video_path = os.path.join(input_directory, filename)
                executor.submit(process_video, video_path)
