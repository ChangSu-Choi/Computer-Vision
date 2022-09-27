import numpy as np
import cv2
import sys

def get_fit(h, w, M):
    """
    실습 코드 채우기
    """
    left_col = sys.maxsize
    right_col = 0
    top_row = sys.maxsize
    bot_row = 0

    move = [(0,0),(h-1,0),(0,w-1),(h-1,w-1)]

    for y, x in move:
        cur = np.array([
            [x],
            [y],
            [1]
        ])

        p = np.dot(M,cur)
        dst_col = p[0][0]
        dst_row = p[1][0]

        left_col = min(left_col, int(dst_col))
        right_col = max(right_col, int(np.ceil(dst_col)))

        top_row = min(top_row, int(dst_row))
        bot_row = max(bot_row, int(np.ceil(dst_row)))

    return left_col, right_col, top_row, bot_row

def forward(src, M):
    print('< forward >')
    print('M')
    print(M)
    h, w = src.shape

    left_col, right_col, top_row, bot_row = get_fit(h, w, M)
    dst = np.zeros((bot_row - top_row +1, right_col - left_col +1))
    dst_h, dst_w = dst.shape
    print('a',dst.shape)
    N = np.zeros(dst.shape)

    for row in range(h):
        for col in range(w):
            P = np.array([[col],[row],[1]])
            P_dst = np.dot(M,P)

            dst_col = P_dst[0][0] - left_col
            dst_row = P_dst[1][0] - top_row

            dst_col_right = int(np.ceil(dst_col))
            dst_col_left = int(dst_col)

            dst_row_bottom = int(np.ceil(dst_row))
            dst_row_top = int(dst_row)

            ## get location 적용 시 어떻게 바꿔야하는가?
            if (0 <= dst_row_top < dst_h and 0 <= dst_col_left < dst_w)\
                and (0 <= dst_row_bottom < dst_h and 0 <= dst_col_right < dst_w):
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
    # box = np.full((50,50), 250, dtype=np.uint8)

    # src[50:100,50:100] = box

    # translation
    M_tr = np.array([
        [1, 0, 50],
        [0, 1, 100],
        [0, 0, 1]
    ])

    #shearing
    M_sh = np.array([
        [1,0.2,0],
        [0.2,1,0],
        [0,0,1]
    ])

    #scaling
    M_sc = np.array([
        [2,0,0],
        [0,2,0],
        [0,0,1]
    ])

    # rotation
    degree = 15
    M_ro = np.array([
        [np.cos(np.deg2rad(degree)), -np.sin(np.deg2rad(degree)), 0],
        [np.sin(np.deg2rad(degree)), np.cos(np.deg2rad(degree)), 0],
        [0, 0, 1]
    ])

    dst_for = forward(src, M_sh)
    # dst_for2 = forward(dst_for, np.linalg.inv(M_sh))

    cv2.imshow('original', src)
    cv2.imshow('forward', dst_for)
    # cv2.imshow('inversM',dst_for2)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ =='__main__':
    main()