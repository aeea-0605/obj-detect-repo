import os, glob
from module.module import *


path = os.getcwd()
video_dir = os.path.join(path, "samples")
video_path = glob.glob(video_dir + "/*9_4.mp4")[0]
video_name = os.path.basename(video_path).split(".")[0]

# Source Video에 대한 정보 객체 생성
info = MakeVideoInfo(video_path)

# Video Open
cap = info.load_video()

# Opened Video에 대한 정보 저장
info.get_attribute(cap)

# Box를 detecting할 target images Load
target_dir = os.path.join(path, 'test_target_imgs')
target_ls = load_target_imgs(target_dir)
print("target image count: " ,len(target_ls))

# 시작할 Frame number 입력
start = int(input(f"insert start frame number (0 ~ {info.count}) : "))
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
    result = detect_box(detect_frame, target_ls)

    idx = np.argmax(result["maxv"])
    th, tw = target_ls[idx].shape[:2]
    src_w, src_h = result["maxloc"][idx]

    # gray_frame에서 선정된 target image의 shape을 따고 그 영역에 대한 tasseract 진행
    # detecting_img = gray_frame[src_h:src_h + th, src_w:src_w + tw]
    # text = detect_text(detecting_img)
    # print(f"Detect :{text}")
    # cv2.imshow('detect_box', detecting_img)
    # cv2.imshow('target', target_ls[idx])

    cv2.rectangle(detect_frame, result["maxloc"][idx], (src_w+tw, src_h+th), (0, 0, 255), 2)
    # cv2.putText(detect_frame, f"{text}", (src_w+30, src_h-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow("src", detect_frame)

    max_value = np.round(result["maxv"][idx], 4)
    print(f"Max Value :{max_value}")

    if max_value <= 0.5829:
        key = cv2.waitKey(0)
        if key == ord('a'):
            cv2.destroyAllWindows()
        
            tmp = np.dstack([gray_frame]*3)

            target_box = make_target_image(tmp, target_dir, inplace=True)
            
            target_ls.append(target_box)

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

    key = cv2.waitKey(info.delay)
    if key == 27:
        print("stop frame number: ", number)
        sys.exit()
    elif key == ord('a'):
        cv2.destroyAllWindows()
        
        tmp = np.dstack([gray_frame]*3)     
        make_target_image(tmp, target_dir)

        cv2.destroyAllWindows()
    elif key  == ord('s') or -1:
        cv2.destroyAllWindows()
    number += 1

cap.release()
cv2.destroyAllWindows()