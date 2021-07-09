import sys
import numpy as np
import cv2

src = cv2.imread('/Users/aeea/Desktop/git/project/side-dl-project/test_imgs/1-20.png')
if src is None:
    print('Image load failed')
    sys.exit()
print(src.shape)

# selectROI를 통한 사각형 영역 설정을 통해 target image 설정
rc = cv2.selectROI(src)
target_img = src[rc[1]:rc[1]+rc[3], rc[0]:rc[0]+rc[2]]


# target image 저장
cv2.imwrite('/Users/aeea/Desktop/git/project/side-dl-project/test_target_imgs/test.png', target_img)
cv2.imshow('target', target_img)

cv2.waitKey()
cv2.destroyAllWindows()