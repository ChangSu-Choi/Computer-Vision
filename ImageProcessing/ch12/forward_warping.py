import numpy as np
import cv2
import sys

def forward(src, M):
    print('< forward >')
    print('M')
    print(M)
    h, w = src.shape

    dst = np.zeros((h, w))
    N = np.zeros(dst.shape)

    for row in range(h):
        for col in range(w):
            P = np.array([
                [col],
                [row],
                [1]
            ])

            P_dst = np.dot(M, P)
            dst_col = P_dst[0][0]
            dst_row = P_dst[1][0]

            dst_col_right = int(np.ceil(dst_col))
            dst_col_left = int(dst_col)

            dst_row_bottom = int(np.ceil(dst_row))
            dst_row_top = int(dst_row)

            if (0 <= dst_row_top < h and 0 <= dst_col_left < w) and (
                    0 <= dst_row_bottom < h and 0 <= dst_col_right < w):
                pass
            else:
                continue

            dst[dst_row_top, dst_col_left] += src[row, col]
            N[dst_row_top, dst_col_left] += 1

            if dst_col_right != dst_col_left:
                dst[dst_row_top, dst_col_right] += src[row, col]
                N[dst_row_top, dst_col_right] += 1

            if dst_row_bottom != dst_row_top:
                dst[dst_row_bottom, dst_col_left] += src[row,col]
                N[dst_row_bottom, dst_col_left] += 1

            if (dst_col_right != dst_col_left) and (dst_row_bottom != dst_row_top):
                dst[dst_row_bottom, dst_col_right] += src[row, col]
                N[dst_row_bottom, dst_col_right] += 1

    dst = np.round(dst / (N + 1E-6))
    dst = dst.astype(np.uint8)
    return dst

def main():
    src = cv2.imread("../imgs/Lena.png", cv2.IMREAD_GRAYSCALE)
    # src = np.zeros((500, 500), dtype=np.uint8)
    box = np.full((50,50), 250, dtype=np.uint8)

    # src[50:100,50:100] = box

    # translation
    M_tr = np.array([
        [1, 0, 50],
        [0, 1, 100],
        [0, 0, 1]
    ])

    # rotation
    degree = 15
    M_ro = np.array([
        [np.cos(np.deg2rad(degree)), -np.sin(np.deg2rad(degree)), 0],
        [np.sin(np.deg2rad(degree)), np.cos(np.deg2rad(degree)), 0],
        [0, 0, 1]
    ])

    dst_for = forward(src, M_ro)

    cv2.imshow('original', src)
    cv2.imshow('forward', dst_for)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ =='__main__':
    main()