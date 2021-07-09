import sys, os, glob
import numpy as np
import cv2


cap = cv2.VideoCapture('/Users/aeea/Desktop/git/project/side-dl-project/samples/ABLATION_1_3.mp4')
# cap = cv2.VideoCapture('/Users/aeea/Desktop/git/project/side-dl-project/samples/ABLATION_9_4.mp4')
if not cap.isOpened():
    print("Video open failed")
    sys.exit()

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
delay = round(1000 / fps)

out = cv2.VideoWriter('/Users/aeea/Desktop/git/project/side-dl-project/result/result_1-3.avi', fourcc, fps, (w, h))
if not out.isOpened():
    print("out open failed!")
    sys.exit()

path = os.getcwd()
target_dir = os.path.join(path, 'test_target_imgs')
target_path = glob.glob(target_dir+'/*.png')

target_ls = []
for path in target_path:
    img = cv2.imread(path)
    if img is None:
        print('target image load failed!')
        sys.exit()

    target_ls.append(img)
print("target image count: " ,len(target_ls))

start = int(input("insert start frame number : "))
number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if number < start:
        number += 1
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = np.dstack([gray_frame]*3)

    result = {"maxv": [], "maxloc": []}
    for i in range(len(target_ls)):
        detect_dst = cv2.matchTemplate(gray_frame, target_ls[i], cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(detect_dst)
        result["maxv"].append(maxv)
        result["maxloc"].append(maxloc)

    target_idx = np.argmax(result["maxv"])
    th, tw = target_ls[target_idx].shape[:2]

    cv2.rectangle(gray_frame, result["maxloc"][target_idx], (result["maxloc"][target_idx][0]+tw, result["maxloc"][target_idx][1]+th), (0, 0, 255), 2)
    
    current_maxv = np.round(result["maxv"][target_idx], 4)
    print(current_maxv)

    cv2.imshow('src', gray_frame)
    cv2.imshow('target', target_ls[target_idx])

    if current_maxv <= 0.5829:
        key = cv2.waitKey()
        if key == ord('a'):
            cv2.destroyAllWindows()
        
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tmp = np.dstack([tmp]*3)
            rc = cv2.selectROI(tmp)
            make_target_img = tmp[rc[1]:rc[1]+rc[3], rc[0]:rc[0]+rc[2]]
            target_ls.append(make_target_img)
            print(len(target_ls))
            cv2.imwrite('/Users/aeea/Desktop/git/project/side-dl-project/test_target_imgs/v2_target{}.png'.format(number), make_target_img)
            cv2.destroyAllWindows()
            number += 1
            continue
        elif key == ord('s'):
            cv2.destroyAllWindows()
            number += 1
            continue
        elif key == 27:
            print(number)
            sys.exit()

    out.write(gray_frame)

    # key = cv2.waitKey(1000)
    key = cv2.waitKey(delay)
    if key == 27:
        print("stop frame number: ", number)
        sys.exit()
    elif key == ord('a'):
        cv2.destroyAllWindows()
        
        tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tmp = np.dstack([tmp]*3)
        rc = cv2.selectROI(tmp)
        make_target_img = tmp[rc[1]:rc[1]+rc[3], rc[0]:rc[0]+rc[2]]
        cv2.imwrite('/Users/aeea/Desktop/git/project/side-dl-project/test_target_imgs/v2_target{}.png'.format(number), make_target_img)
        cv2.destroyAllWindows()
    elif key  == ord('s') or -1:
        cv2.destroyAllWindows()
    number += 1

out.release()
cap.release()
cv2.destroyAllWindows()
