import sys
import os, glob
import numpy as np
import cv2

# source, target image들의 path를 구함
path = os.getcwd()
imgs_dir = os.path.join(path, 'test_imgs')
imgs_path = glob.glob(imgs_dir+'/*.png')
target_dir = os.path.join(path, 'test_target_imgs')
target_path = glob.glob(target_dir+'/*.png')

# matchTemplate에 사용될 target image 리스트 생성
target_ls = []
for path in target_path:
    img = cv2.imread(path)
    target_ls.append(img)

# 모든 source image에 대한 detecting code 실행
for i in range(len(imgs_path)):
    src = cv2.imread(imgs_path[i])
    if src is None:
        print("src load failed!")
        sys.exit()
    
    # 각 source image에 대해 target_ls에 있는 모든 target image에 대해 matchTemplate을 하고 그에 대한 max value, max location을 append
    result_dict = {"maxv": [], "maxloc": []}
    for num in range(len(target_ls)):
        result = cv2.matchTemplate(src, target_ls[num], cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(result)
        result_dict["maxv"].append(maxv)
        result_dict["maxloc"].append(maxloc)

    # target image들에 대한 max value 중 가장 높은 max value에 대한 index 도출(source image에 대한 target image 선정)
    target_idx = np.argmax(result_dict["maxv"])

    # 선정된 target image의 height, width 도출
    th, tw = target_ls[target_idx].shape[:2]

    # source image에서 detected된 영역 사각형으록 그리기
    cv2.rectangle(src, result_dict["maxloc"][target_idx], (result_dict["maxloc"][target_idx][0]+tw, result_dict["maxloc"][target_idx][1]+th), (0, 0, 255), 2)

    print(np.round(result_dict["maxv"][target_idx], 4))
    cv2.imshow('src', src)
    cv2.imshow('target', target_ls[target_idx])

    # s를 누르면 다음 source image로 넘어가고, esc를 누르면 code 종료
    # targeting이 되지 않은 이미지가 뜨면 esc를 눌러 해당 이미지의 경로를 확인해 extract_target.py에서 해당 이미지에 대한 target image를 생성했음
    key = cv2.waitKey(0)
    if key == 27:
        print(imgs_path[i])
        sys.exit()
    elif key  == ord('s'):
        cv2.destroyAllWindows()

cv2.destroyAllWindows()