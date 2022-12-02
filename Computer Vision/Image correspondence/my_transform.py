"""
2022년 DSC공유대학 computer vision
"""
import numpy as np
import cv2

def transformation_backward(src, M):
    h, w, c = src.shape
    rate = 2
    dst = np.zeros((int(h * rate), int(w * rate), c))

    h_, w_ = dst.shape[:2]

    print("backward calc")
    for row_ in range(h_):
        for col_ in range(w_):
            xy = (np.linalg.inv(M)).dot(np.array([[col_, row_, 1]]).T)
            x = xy[0, 0]
            y = xy[1, 0]

            # pixel의 값을 가져오기 위해서 bilinear 연산을 통해서 값을 가져옴
            floorX = int(x)
            floorY = int(y)

            t, s = x - floorX, y- floorY

            zz = (1-t) * (1-s)
            zo = t * (1-s)
            oz = (1-t) * s
            oo = t * s

            if floorY < 0 or floorX < 0 or (floorY + 1) >= h or (floorX + 1) >= w:
                continue

            interVal = src[floorY, floorX, :] * zz + src[floorY, floorX + 1, :] * zo +\
                       src[floorY + 1, floorX, :] * oz + src[floorX + 1, floorX + 1, :] * oo

            dst[row_, col_, :] = interVal

    dst = ((dst - np.min(dst)) / np.max(dst - np.min(dst)) * 255 + 0.5)  # normalization
    return dst.astype(np.uint8)


def transformation_forward(src, M):
    h, w, c = src.shape
    rate = 2  # 변환을 생각하여 임의로 크기를 키움
    dst = np.zeros((int(h * rate), int(w * rate), c))

    h_, w_ = dst.shape[:2]
    count = dst.copy()

    print("forward calc")
    for row in range(h):
        for col in range(w):
            xy_prime = np.dot(M, np.array([[col, row, 1]]).T)
            x_ = xy_prime[0, 0]
            y_ = xy_prime[1, 0]

            if x_ < 0 or y_ < 0 or x_ >= w_ or y_ >= h_:
                # 벗어나는 범위에 대해서 예외 처리
                continue

            dst[int(y_), int(x_), :] += src[row, col, :] # 얻은 값을 누적
            count[int(y_), int(x_), :] += 1              # 동일한 위치에서 누적되는 값을 처리하기 위함

    dst = (dst/count)
    return dst.astype(np.uint8)


def main():
    src = cv2.imread("../imgs/Lena.png")
    src = cv2.resize(src, None, fx=0.3, fy=0.3)
    theta = -10

    translation = [[1, 0, 350],
                   [0, 1, 350],
                   [0, 0, 1]]
    rotation = [[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]]
    scaling = [[2, 0, 0],
               [0, 2, 0],
               [0, 0, 1]]

    M = np.dot(np.dot(translation, rotation), scaling)

    forward = transformation_forward(src, M)
    backward = transformation_backward(src, M)

    cv2.imshow("forward", forward)
    cv2.imshow("backward", backward)
    cv2.imwrite("./results/forward.png", forward)
    cv2.imwrite("./results/backward.png", backward)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()