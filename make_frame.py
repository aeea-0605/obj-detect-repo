import os
import glob
import cv2
import numpy as np


path = os.getcwd()
download_path = os.path.join(path, 'test_imgs')

videos = glob.glob(path+"/samples"+"/*.mp4")
if videos == []:
    print("No video")
    sys.exit()

for idx, video in enumerate(videos):

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"Video{idx} open failed!")
        sys.exit()

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stride = int(np.ceil(length / 15))
    for num in range(length):
        ret, frame = cap.read()
        if not ret:
            break
        
        if num % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(download_path+f"/{idx}-{num}.png", frame, params=[cv2.IMWRITE_PNG_COMPRESSION,0])

    cap.release()
    print(f"End {idx}")

cv2.destroyAllWindows()