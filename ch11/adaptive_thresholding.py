import cv2
import numpy as np

def adaptive_threshold(img, group_num=4):
    img_split = np.hsplit(img, group_num)

    thresholds = list()
    dst = list()

    for img_part in img_split:
        val, dst_part = cv2.threshold(img_part, 0, 255, cv2.THRESH_OTSU)
        thresholds.append(val)
        dst.append(dst_part)

    dst = np.concatenate(dst, axis=1)
    return dst, thresholds


if __name__ == '__main__':
    img = cv2.imread('../imgs/circles_adaptive_threshold.png', cv2.IMREAD_GRAYSCALE)

    dst, val = adaptive_threshold(img, group_num=4)
    print('Threshold: ', val)

    cv2.imshow('original', img)
    cv2.imshow('adaptive threshold', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


