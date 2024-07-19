import cv2
import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def extract_frames(video_file, fps, duration, output_directory, repeat):
    fps = 10  # Desired frame rate
    frame_count = 0
    image_count = 0
    total_images = fps * duration
    os.makedirs(output_directory, exist_ok=True)
    video_file = video_file 
    cap = cv2.VideoCapture(video_file)
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        frame_count += 1
        if image_count < total_images:
            # Only extract frames at the desired frame rate
            if frame_count % int(cap.get(5) / fps) == 0:
                ## check if the total number of images has been reached if yes break
                output_file = f"{output_directory}/repeatation{repeat}_{image_count}.jpg"
                # save the image
                cv2.imwrite(output_file, frame)
                print(f"Frame {frame_count} has been extracted and saved as {output_file}")
                image_count+=1
        else:
            print("Exceeded video duration!!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    ## --> Test case
    # dir_name = "test_videos"
    # out_dirName = "test"
    ## --> Train case
    dir_name = "train_videos"
    out_dirName = "train"

    for test_folders in sorted(os.listdir(dir_name)):
        objectsInFolder = len(os.listdir(f"{dir_name}/{test_folders}"))
        for i in range(objectsInFolder+1):
            extract_frames(f"{dir_name}/{test_folders}/0 ({i}).mp4", 10, 15,
                        f"customDataset/{out_dirName}/{test_folders}", i, )