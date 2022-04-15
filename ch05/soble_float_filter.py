import cv2
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_filtering_true import my_filtering

def get_sobel():
    derivateive = np.array([[-1, 0, 1]])
    blur = np.array([[1], [2], [1]])

    x = np.dot(blur, derivateive)
    y = np.dot(derivateive.T, blur.T)

    return x, y

def main():
    sobel_x, sobel_y = get_sobel()

    src = cv2.imread("../imgs/edge_detection_img.png", cv2.IMREAD_GRAYSCALE)
    dst_x = my_filtering(src, sobel_x)
    dst_y = my_filtering(src, sobel_y)

    dst_x = np.sqrt(dst_x ** 2)
    dst_y = np.sqrt(dst_y ** 2)

    dst = dst_x + dst_y

    dst_x_norm = (dst_x - np.min(dst_x)) / np.max(dst_x - np.min(dst_x))
    dst_y_norm = (dst_y - np.min(dst_y)) / np.max(dst_y - np.min(dst_y))

    cv2.imshow("dst_x", dst_x)
    cv2.imshow("dst_y", dst_y)
    cv2.imshow("dst_x_norm", dst_x_norm)
    cv2.imshow("dst_y_norm", dst_y_norm)
    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

