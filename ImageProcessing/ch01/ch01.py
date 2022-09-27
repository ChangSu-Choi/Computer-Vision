import cv2
import matplotlib.pyplot as plt
import numpy as np




def my_BGR2GRAY(img):
    # cvtColor 함수 사용
    bgr_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cvtColor로 gray scale 구현

    # B, G, R 긱각 1/3씩 사용
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    div_x = b/3+g/3+r/3 # 미리 더한다음 나누면 255초과로 정보 손

    # B, G, R 중 하나 선택
    bgr_select = img[:, :, 0]  # B channel 선택
    # 변환 공식 사용
    # B*0.0721 + G*0.7154 + R*0.2125 로 계산
    bgr_phosphor = ((0.0721 * b) + (0.7154 * g) + (0.2125 * r))

    return bgr_cvt, div_x, bgr_select, bgr_phosphor

if __name__ == '__main__':
    img = cv2.imread('../imgs/fruits.jpg')
    cv2.imshow('img', img)
    bgr_cvt, div_x, bgr_select, bgr_phosphor = my_BGR2GRAY(img)

    cv2.imshow('bgr_cvt', bgr_cvt.astype(np.uint8))
    cv2.imwrite('result/cvtColor.jpg', bgr_cvt.astype(np.uint8))

    cv2.imshow('div_x', div_x.astype(np.uint8))
    cv2.imwrite('result/B_3+R_3+G_3.jpg', div_x.astype(np.uint8))

    cv2.imshow('bgr_select', bgr_select.astype(np.uint8))
    cv2.imwrite('result/B,G,R중 하나 선택.jpg', bgr_select.astype(np.uint8))

    cv2.imshow('bgr_phosphor', bgr_phosphor.astype(np.uint8))
    cv2.imwrite('result/변환공식사용.jpg', bgr_phosphor.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
