import cv2
import numpy as np
import cv2 as cv


def my_padding(src, pad_shape):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:p_h + h, p_w:p_w + w] = src

    return pad_img


def dilation(B, S):
    (b_h, b_w) = B.shape
    (s_h, s_w) = S.shape
    pad_size_h = s_h // 2
    pad_size_w = s_w // 2

    dst = my_padding(B, (pad_size_h, pad_size_w))
    ###############################################
    # TODO                                        #
    # dilation 함수 완성                           #
    ###############################################

    sum = np.sum(S)
    if sum == 9:
        # 8 neighbor
        for row in range(b_h):
            for col in range(b_w):
                if B[row][col] == 1:
                    dst[row:row+s_h, col:col+s_w] = 1
    elif sum == 5:
        # 4 neighbor
        for row in range(b_h):
            dst_row = row + pad_size_h
            for col in range(b_w):
                dst_col = col + pad_size_w
                if B[row][col] == 1:
                    dst[dst_row-pad_size_h:dst_row + pad_size_h+1, dst_col] = 1
                    dst[dst_row, dst_col - pad_size_w: dst_col + pad_size_w+1] = 1

    dst = dst[pad_size_h:b_h + pad_size_h, pad_size_w:b_w + pad_size_w]
    return dst


def erosion(B, S):
    (b_h, b_w) = B.shape
    (s_h, s_w) = S.shape
    pad_size_h = s_h // 2
    pad_size_w = s_w // 2

    dst = np.zeros(B.shape)
    ###############################################
    # TODO                                        #
    # erosion 함수 완성                            #
    ###############################################
    sum = np.sum(S)
    if sum == 9:
        # 8 neighbor
        for row in range(pad_size_h, b_h-pad_size_h):
            for col in range(pad_size_w, b_w-pad_size_h):
                if np.array_equal(B[row-pad_size_h:row+pad_size_h+1, col-pad_size_h:col+pad_size_w+1], S):
                    dst[row][col] = 1

    elif sum == 5:
        # 4 neighbor
        for row in range(pad_size_h, b_h - pad_size_h):
            for col in range(pad_size_w, b_w - pad_size_h):
                # X축과 Y축 비교
                if np.array_equal(B[row - pad_size_h:row + pad_size_h + 1, col], S[1,:]) and \
                        np.array_equal(B[row, col - pad_size_h:col + pad_size_w + 1], S[:,1]):
                    dst[row][col] = 1

    return dst


def opening(B, S):
    ###############################################
    # TODO                                        #
    # opening 함수 완성                            #
    ###############################################
    dst = erosion(B, S)
    dst = dilation(dst, S)

    return dst


def closing(B, S):
    ###############################################
    # TODO                                        #
    # closing 함수 완성                            #
    ###############################################
    dst = dilation(B, S)
    dst = erosion(dst, S)
    return dst


if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    # S = np.array(
    #     [[0, 1, 0],
    #      [1, 1, 1],
    #      [0, 1, 0]])

    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])

    cv2.imwrite('./result/morphology_B_8.png', (B * 255).astype(np.uint8))

    img_dilation = dilation(B, S)
    img_dilation = (img_dilation * 255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('./result/morphology_dilation_8.png', img_dilation)

    img_erosion = erosion(B, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('./result/morphology_erosion_8.png', img_erosion)

    img_opening = opening(B, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('./result/morphology_opening_8.png', img_opening)

    img_closing = closing(B, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('./result/morphology_closing_8.png', img_closing)
