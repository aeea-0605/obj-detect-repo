import sys, glob
import cv2
import numpy as np
import pytesseract
import re


def detect_text(image, blur_size=5, h_rate=0.025, w_rate=0.065, bin_value=93.5):
    
    # 임계값 도출 > 그 값을 기준으로 흑백 이진화 진행 후 모서리 영역에 대한 픽셀값 255로 변경
    classify_value = np.percentile(image.flatten(), bin_value)
    image_bin = cv2.threshold(image, classify_value, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    image_bin[:, :int(np.round(image.shape[1] * w_rate))] = 255
    image_bin[:int(np.round(image.shape[0] * h_rate)), :] = 255

    # 메디안 블러를 통해 text에 대한 잡음제거
    blur_img = cv2.medianBlur(image_bin, blur_size)

    text = pytesseract.image_to_string(blur_img, config='--psm 6', lang=None)

    result =  re.findall("(\d+|N/A)", text)
    if "N/A" in result:
        detected_text = "N/A"
        # print("N/A", target_names[i].split(".")[0].split("_")[0])
    elif result == []:
        detected_text = "detect failed"
        # print("None", target_names[i].split(".")[0].split("_")[0])
    else:
        detected_text = "".join(result)
        # print("".join(result), target_names[i].split(".")[0].split("_")[0])

    return detected_text


class MakeVideoInfo:

    def __init__(self, path):
        self.path = path
        self.w = None
        self.h = None
        self.count = None
        self.fps = None
        self.fourcc = None
        self.delay = None

    
    def load_video(self):
        video = cv2.VideoCapture(self.path)
        if not video.isOpened():
            print("Video Load failed!!")
            sys.exit()

        return video


    def get_attribute(self, video):
        self.w = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.delay = round(1000 / self.fps)


    def load_save_video(self, save_path):
        save_video = cv2.VideoWriter(f"{save_path}.avi", self.fourcc, self.fps, (self.w, self.h))
        if not save_video.isOpened():
            print("Save Video Load failed!!")
            sys.exit()

        return save_video



def load_target_imgs(path):
    target_path = glob.glob(path+'/*.png')

    target_ls = []
    for path in  target_path:
        image = cv2.imread(path)
        if image is None:
            print("Target Image Load failed!!")
            sys.exit()

        target_ls.append(image)

    return target_ls


def detect_box(frame, target_ls):
    result = {"maxv": [], "maxloc": []}
    for i in range(len(target_ls)):
        detect_img = cv2.matchTemplate(frame, target_ls[i], cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(detect_img)
        result["maxv"].append(maxv)
        result["maxloc"].append(maxloc)

    return result
    

def make_target_image(frame, save_path, inplace=False):
    rc = cv2.selectROI(tmp)
    target_box = tmp[rc[1]:rc[1]+rc[3], rc[0]:rc[0]+rc[2]]

    cv2.imwrite(f'{save_path}/tmp_target.png', target_box)

    if inplace:
        return target_box

        