import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_threshold(img, th=120):
    ######################################################
    # TODO                                               #
    # 실습시간에 배포된 코드 사용                             #
    ######################################################
    dst = np.zeros(img.shape, img.dtype)
    dst[img >= th] = 255
    dst[img < th] = 0
    return dst


def my_otsu_threshold(img):
    hist, bins = np.histogram(img.ravel(),256,[0,256])
    p = hist / np.sum(hist) + 1e-7
    ######################################################
    # TODO                                               #
    # Otsu 방법을 통해 threshold 구한 후 이진화 수행          #
    # cv2의 threshold 와 같은 값이 나와야 함                 #
    ######################################################
    hist = hist / np.sum(hist)
    val_max = -99999
    th = 0
    try:
        for t in range(0, 256):
            q1 = np.sum(hist[:t])
            q2 = np.sum(hist[t:])
            m1 = np.sum(np.array([i for i in range(t)]) * hist[:t]) / q1
            m2 = np.sum(np.array([i for i in range(t, 256)]) * hist[t:]) / q2
            val = q1 * (1 - q1) * (m1 - m2)**2
            if val_max < val:
                val_max = val
                th = t
    except:
        pass

    # th = np.argmax(temp)
    dst = apply_threshold(img, th)

    return th, dst

if __name__ == '__main__':
    img = cv2.imread('../imgs/cameraman.tif', cv2.IMREAD_GRAYSCALE)

    th_cv2, dst_cv2 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    th_my, dst_my = my_otsu_threshold(img)

    print('Threshold from cv2: {}'.format(th_cv2))
    print('Threshold from my: {}'.format(th_my))

    cv2.imshow('original image 20181602 CCS', img)
    cv2.imshow('cv2 threshold 20181602 CCS', dst_cv2)
    cv2.imshow('my threshold 20181602 CCS', dst_my)
    cv2.imwrite('./result/original image.png', img)
    cv2.imwrite('./result/cv2 threshold.png', dst_cv2)
    cv2.imwrite('./result/my threshold.png', dst_my)

    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


