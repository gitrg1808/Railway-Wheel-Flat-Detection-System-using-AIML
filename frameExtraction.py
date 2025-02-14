
#frame extraction along with axle counter

import cv2
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np

def extract_unique_frames(video_path, output_folder, similarity_threshold=0.95):
  
    # creating output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # opening video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Unable to open video.")
        return

    frame_id = 0
    saved_frame_id = 0
    last_saved_frame = None
    axle_counter = 1  # Start with axle 1
    wheel_in_axle = 0  # To keep track of two wheels in one axle

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # conversion to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if last_saved_frame is None:
            
            filename = os.path.join(output_folder, f"wheel_{saved_frame_id:02d}_axle{axle_counter}.jpg")
            cv2.imwrite(filename, frame)
            last_saved_frame = gray_frame
            saved_frame_id += 1
            wheel_in_axle += 1
        else:
            # SSIM calculation between the current frame and the last saved frame
            
            score, _ = ssim(last_saved_frame, gray_frame, full=True)
            if score < similarity_threshold:
                # Save the frame if it is not similar to the last saved frame
                filename = os.path.join(output_folder, f"wheel_{saved_frame_id:02d}_axle{axle_counter}.jpg")
                cv2.imwrite(filename, frame)
                last_saved_frame = gray_frame
                saved_frame_id += 1
                wheel_in_axle += 1

        # axle number update
        if wheel_in_axle == 2:
            axle_counter += 1
            wheel_in_axle = 0

        frame_id += 1

    video.release()
    print(f"Extraction completed. {saved_frame_id} unique frames saved to {output_folder}.")

video_path = "D:/KP/Wabtech/flat_test5.mp4"
output_folder = "D:/KP/Wabtech/ExtractedImg"
extract_unique_frames(video_path, output_folder, similarity_threshold=0.75)
