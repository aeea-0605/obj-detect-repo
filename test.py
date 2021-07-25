import sys, os, glob
from module.module import *
from detect_tesseract import *


path = os.getcwd()

# Source Video 열기
cap = cv2.VideoCapture('/Users/aeea/Desktop/git/project/obj-detect-repo/samples/ABLATION_9_4.mp4')
if not cap.isOpened():
    print("Video open failed")
    sys.exit()

# Source Video의 속성
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
delay = round(1000 / fps)

# 결과를 저장할 객체 생성
out = cv2.VideoWriter('/Users/aeea/Desktop/git/project/obj-detect-repo/result/result_9-4-2.avi', fourcc, fps, (w, h))
if not out.isOpened():
    print("out open failed!")
    sys.exit()

# Target Image Load
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

# 시작할 Frame number 입력
start = int(input(f"insert start frame number : 0 ~ {count} >>"))
number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if number < start:
        number += 1
        continue

    # 현재 frame 그레이스케일 후 target image와 비교를 위한 demension 조정
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_frame = np.dstack([gray_frame]*3)

    # 모든 target image들과 matchTemplate후 가장 높은 value를 가진 target으로 detecting
    result = {"maxv": [], "maxloc": []}
    for i in range(len(target_ls)):
        detect_dst = cv2.matchTemplate(detect_frame, target_ls[i], cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(detect_dst)
        result["maxv"].append(maxv)
        result["maxloc"].append(maxloc)

    target_idx = np.argmax(result["maxv"])
    th, tw = target_ls[target_idx].shape[:2]
    src_w, src_h = result["maxloc"][target_idx]

    # 여기서 이미지 추출 > gray_frame에서 선정된 target image의 shape을 따고 그 영역에 대한 tasseract 진행
    detecting_img = gray_frame[src_h:src_h + th, src_w:src_w + tw]
    text = detect_text(detecting_img)
    print(f"Detect :{text}")
    cv2.imshow('detect_box', detecting_img)
    cv2.imshow('target', target_ls[target_idx])

    cv2.rectangle(detect_frame, result["maxloc"][target_idx], (src_w+tw, src_h+th), (0, 0, 255), 2)
    cv2.putText(detect_frame, f"{text}", (src_w+30, src_h-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow("src", detect_frame)

    current_maxv = np.round(result["maxv"][target_idx], 4)
    print(f"Max Value :{current_maxv}")

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
            cv2.imwrite('/Users/aeea/Desktop/git/project/obj-detect-repo/test_target_imgs/tmp_target{}.png'.format(number), make_target_img)
            cv2.destroyAllWindows()
            number += 1
            continue
        elif key == ord('s'):
            out.write(detect_frame)
            cv2.destroyAllWindows()
            number += 1
            continue
        elif key == 27:
            print(number)
            sys.exit()

    out.write(detect_frame)

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
        cv2.imwrite('/Users/aeea/Desktop/git/project/side-dl-project/test_target_imgs/tmp_target{}.png'.format(number), make_target_img)
        cv2.destroyAllWindows()
    elif key  == ord('s') or -1:
        cv2.destroyAllWindows()
    number += 1

out.release()
cap.release()
cv2.destroyAllWindows()
