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
    src = cv2.imread("../result/building.jpg", cv2.IMREAD_GRAYSCALE)
    dst_x = my_filtering(src, sobel_x)
    dst_y = my_filtering(src, sobel_y)

    # dst_x = np.clip(dst_x, 0, 255).astype(np.uint8)
    # dst_y = np.clip(dst_y, 0, 255).astype(np.uint8)

    dst_x = np.abs(dst_x)
    dst_y = np.abs(dst_y)

    cv2.imshow("dst_x", dst_x/255)
    cv2.imshow("dst_y", dst_y/255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()