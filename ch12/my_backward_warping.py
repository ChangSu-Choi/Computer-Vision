import numpy as np
import cv2
import sys

def get_fit(h, w, M):
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


def backward(src, M):
    #####################################################
    # TODO                                              #
    # backward 완성                                      #
    #####################################################
    print('< backward >')
    print('M')
    print(M)
    h, w = src.shape
    M_inv = np.linalg.inv(M)
    print('M inv')
    print(M_inv)

    left_col, right_col, top_row, bot_row = get_fit(h, w, M)
    dst = np.zeros((bot_row - top_row+1, right_col - left_col+1))

    print('dst.shape', dst.shape)
    print('src.shape', src.shape)
    print('right_col - left_col', right_col - left_col)
    print('top_row - bot_row', bot_row - top_row)
    for row in range(top_row, bot_row):
        for col in range(left_col, right_col):

            # print('row: ', row)
            # print('col: ', col)

            P_dst = np.array([[col],[row],[1]])
            P = np.dot(M_inv,P_dst)

            src_col = P[0][0]
            src_row = P[1][0]

            src_col_right = int(np.ceil(src_col))
            src_col_left = int(src_col)

            src_row_bottom = int(np.ceil(src_row))
            src_row_top = int(src_row)

            ######################################################
            # TODO                                               #
            # 범위가 벗어 나는 경우에 대한 예외 처리 해주기             #
            ######################################################
            if (0 <= src_row_top < h and 0 <= src_col_left < w)\
                and (0 <= src_row_bottom < h and 0 <= src_col_right < w):
                pass
            else:
                continue


            ######################################################
            # TODO                                               #
            # bilinear을 사용하여 backward 완성하기                 #
            ######################################################

            m = int(src_row)
            n = int(src_col)
            t = -(m-src_row)
            s = -(n-src_col)

            try:
                if m+1 > h-1  and n+1 > w-1:
                    intensity = (1-s)*(1-t)*src[m][n]+s*(1-t)*src[m][n]+(1-s)*t*src[m][n]+s*t*src[m][n]
                elif m+1 > h-1:
                    intensity = (1-s)*(1-t)*src[m][n]+s*(1-t)*src[m][n+1]+(1-s)*t*src[m][n]+s*t*src[m][n+1]
                elif n+1 > w-1:
                    intensity = (1-s)*(1-t)*src[m][n]+s*(1-t)*src[m][n]+(1-s)*t*src[m+1][n]+s*t*src[m+1][n]
                else:
                    intensity = (1-s)*(1-t)*src[m][n]+s*(1-t)*src[m][n+1]+(1-s)*t*src[m+1][n]+s*t*src[m+1][n+1]
            except:
                pass

            # print('col-left_col: ', col-left_col)
            dst[row-top_row, col-left_col] = intensity


    # dst = np.round(dst / (N + 1E-6))
    dst = dst.astype(np.uint8)
    return dst

def main():
    src = cv2.imread("../imgs/Lena.png", cv2.IMREAD_GRAYSCALE)

    # translation
    M_tr = np.array([
        [1, 0, 50],
        [0, 1, 100],
        [0, 0, 1]
    ])

    # scaling
    M_sc = np.array([
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 1]
    ])

    # shearing
    M_sh = np.array([
        [1, 0.2, 0],
        [0.2, 1, 0],
        [0, 0, 1]
    ])

    # rotation
    degree = 15
    M_ro = np.array([
        [np.cos(np.deg2rad(degree)), -np.sin(np.deg2rad(degree)), 0],
        [np.sin(np.deg2rad(degree)), np.cos(np.deg2rad(degree)), 0],
        [0, 0, 1]
    ])

    M = np.dot(np.dot(np.dot(M_sh, M_sc), M_tr), M_ro)
    # M = np.dot(M_ro)

    dst_for = backward(src, M)
    dst_for2 = backward(dst_for, np.linalg.inv(M))

    cv2.imshow('20181602_original', src)
    # 출력 결과에 본인 학번이 나오도록 작성
    cv2.imshow('20181602_backward', dst_for)
    cv2.imshow('20181602_backward2', dst_for2)
    cv2.imwrite('./result/original.jpg', src)
    cv2.imwrite('./result/backward.jpg', dst_for)
    cv2.imwrite('./result/backward2.jpg', dst_for2)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ =='__main__':
    main()