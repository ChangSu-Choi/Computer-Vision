import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('../imgs/fruits.jpg')
cv2.imshow('img', img)


def my_BGR2GRAY(img):
    # cvtColor 함수 사용
    bgr_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cvtColor로 gray scale 구현

    # B, G, R 긱각 1/3씩 사용
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    div_x = (b+g+r)/3

    # B, G, R 중 하나 선택
    bgr_select = img[:, :, 0]  # B channel 선택
    print(bgr_select)
    # 변환 공식 사용
    # B*0.0721 + G*0.7154 + R*0.2125 로 계산
    bgr_phosphor = ((0.0721 * b) + (0.7154 * g) + (0.2125 * r))

    return bgr_cvt, div_x, bgr_select, bgr_phosphor


bgr_cvt, div_x, bgr_select, bgr_phosphor = my_BGR2GRAY(img)

cv2.imshow('bgr_cvt', bgr_cvt.astype(np.uint8))
cv2.imwrite('./result/cvtColor.jpg', bgr_cvt.astype(np.uint8))

cv2.imshow('div_x', div_x.astype(np.uint8))
cv2.imwrite('./result/B_3+R_3+G_3.jpg', div_x.astype(np.uint8))

cv2.imshow('bgr_select', bgr_select.astype(np.uint8))
cv2.imwrite('./result/B,G,R중 하나 선택.jpg', bgr_select.astype(np.uint8))

cv2.imshow('bgr_phosphor', bgr_phosphor.astype(np.uint8))
cv2.imwrite('./result/변환공식사용.jpg', bgr_phosphor.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()
