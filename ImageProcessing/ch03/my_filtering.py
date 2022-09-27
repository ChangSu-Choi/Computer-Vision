import cv2
import numpy as np


def my_get_Gaussian2D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 2D gaussian filter 만들기
    #########################################
    h, w = msize
    y, x = np.mgrid[-(h // 2):(h // 2) + 1, -(w // 2):(w // 2) + 1]
    print('y', y, 'x', x)

    '''
    y, x = np.mgrid[-1:2, -1:2]
    y = [[-1,-1,-1],
         [ 0, 0, 0],
         [ 1, 1, 1]]
    x = [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]
    '''
    # 파이 => np.pi 를 쓰시면 됩니다.
    # 2차 gaussian mask 생성
    gaus2D = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)

    return gaus2D


def my_get_Gaussian1D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 1D gaussian filter 만들기
    #########################################
    # 메인에 y 트랜스포즈
    x = np.full((1, msize), [range(-(msize // 2), (msize // 2) + 1)])
    '''
    x = np.full((1, 3), [-1, 0, 1])
    x = [[ -1, 0, 1]]

    x = np.array([[-1, 0, 1]])
    x = [[ -1, 0, 1]]
    '''

    # 파이 => np.pi 를 쓰시면 됩니다.
    gaus1D = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-x ** 2 / (2 * sigma))

    # mask의 총 합 = 1
    gaus1D /= np.sum(gaus1D)
    return gaus1D


def my_mask(ftype, fshape, sigma=1):
    if ftype == 'average':
        print('average filtering')
        ###################################################
        # TODO                                            #
        # mask 완성                                       #
        ###################################################
        h, w = fshape
        mask = np.ones(fshape) / (w * h)

        # mask 확인
        print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        ##################################################
        # TODO                                           #
        # mask 완성                                      #
        ##################################################
        w, h = fshape
        base_mask = np.zeros(fshape)
        base_mask[fshape[0] // 2, fshape[1] // 2] = 2
        aver_mask = np.ones(fshape) / (w * h)
        mask = base_mask - aver_mask

        # mask 확인
        print(mask)

    elif ftype == 'gaussian2D':
        print('gaussian filtering 2D')
        ##################################################
        # TODO                                           #
        # mask 완성                                      #
        ##################################################
        mask = my_get_Gaussian2D_mask(fshape, sigma=sigma)
        # mask 확인
        print(mask)

    elif ftype == 'gaussian1D':
        print('gaussian filtering 1D')
        ##################################################
        # TODO                                           #
        # mask 완성                                      #
        ##################################################
        mask = my_get_Gaussian1D_mask(fshape, sigma=sigma)
        # mask 확인
        print(mask)

    return mask


def my_zero_padding(src, pad_shape):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:p_h + h, p_w:p_w + w] = src
    return pad_img


def my_filtering(src, mask):
    #########################################################
    # TODO                                                  #
    # dst 완성                                              #
    # dst : filtering 결과 image                            #
    #########################################################
    h, w = src.shape
    m_h, m_w = mask.shape
    pad_img = my_zero_padding(src, (m_h // 2, m_w // 2))
    dst = np.zeros((h, w))

    """
    반복문을 이용하여 filtering을 완성하기
    """
    for row in range(h):
        # src_row = row + m_h // 2
        for col in range(w):
            # src_col = col + m_w // 2
            # val = np.sum(np.multiply(mask, pad_img[src_row - (m_h // 2):src_row + (m_h // 2) + 1,src_col - (m_w // 2): src_col + (m_w // 2) + 1]))
            val = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)
            val = np.clip(val, 0, 255)  # 범위를 0~255로 조정
            dst[row][col] = val

    dst = (dst + 0.5).astype(np.uint8)  # uint8의 형태로 조정

    return dst


if __name__ == '__main__':
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)

    # 3x3 filter
    average_mask = my_mask('average', (3, 3))
    sharpening_mask = my_mask('sharpening', (3, 3))

    # 원하는 크기로 설정
    # dst_average = my_filtering(src, 'average', (5,5))
    # dst_sharpening = my_filtering(src, 'sharpening', (5,5))

    # 11x13 filter
    # dst_average = my_filtering(src, 'average', (5,3), 'repetition')
    # dst_sharpening = my_filtering(src, 'sharpening', (5,3), 'repetition')
    # dst_average = my_filtering(src, 'average', (11,13))
    # dst_sharpening = my_filtering(src, 'sharpening', (11,13))

    cv2.imshow('original', src)
    cv2.imwrite('./result/입력 이미지.jpg', src)

    # average filtering
    average_mask_3by3 = my_mask('average', (3, 3))
    average_mask_3by5 = my_mask('average', (3, 5))
    average_mask_5by5 = my_mask('average', (5, 5))

    dst_average_3by3 = my_filtering(src, average_mask_3by3)
    dst_average_3by5 = my_filtering(src, average_mask_3by5)
    dst_average_5by5 = my_filtering(src, average_mask_5by5)

    # cv2.imshow('average filter 3by3', dst_average_3by3)
    cv2.imwrite('./result/average filter 3by3.jpg', dst_average_3by3)
    # cv2.imshow('average filter 3by5', dst_average_3by5)
    cv2.imwrite('./result/average filter 3by5.jpg', dst_average_3by5)
    # cv2.imshow('average filter 5by5', dst_average_5by5)
    cv2.imwrite('./result/average filter 5by5.jpg', dst_average_5by5)

    # sharpening filtering
    sharpening_mask_3by3 = my_mask('sharpening', (3, 3))
    sharpening_mask_3by5 = my_mask('sharpening', (3, 5))
    sharpening_mask_5by5 = my_mask('sharpening', (5, 5))

    dst_sharpening_3by3 = my_filtering(src, sharpening_mask_3by3)
    dst_sharpening_3by5 = my_filtering(src, sharpening_mask_3by5)
    dst_sharpening_5by5 = my_filtering(src, sharpening_mask_5by5)

    # cv2.imshow('sharpening filter 3by3', dst_sharpening_3by3)
    cv2.imwrite('./result/sharpening filter 3by3.jpg', dst_sharpening_3by3)
    # cv2.imshow('sharpening filter 3by5', dst_sharpening_3by5)
    cv2.imwrite('./result/sharpening filter 3by5.jpg', dst_sharpening_3by5)
    # cv2.imshow('sharpening filter 5by5', dst_sharpening_5by5)
    cv2.imwrite('./result/sharpening filter 5by5.jpg', dst_sharpening_5by5)

    # 2D Gaussian filter
    gaussian2d_mask_3by3_sig1 = my_mask('gaussian2D', (3, 3), sigma=1)
    gaussian2d_mask_5by5_sig1 = my_mask('gaussian2D', (5, 5), sigma=1)
    gaussian2d_mask_7by7_sig1 = my_mask('gaussian2D', (7, 7), sigma=1)

    gaussian2d_mask_3by3_sig_dot5 = my_mask('gaussian2D', (3, 3), sigma=0.5)
    gaussian2d_mask_5by5_sig3 = my_mask('gaussian2D', (5, 5), sigma=3)
    gaussian2d_mask_7by7_sig_dot1 = my_mask('gaussian2D', (7, 7), sigma=0.1)


    dst_gaussian2d_3by3_sig1 = my_filtering(src, gaussian2d_mask_3by3_sig1)
    dst_gaussian2d_5by5_sig1 = my_filtering(src, gaussian2d_mask_5by5_sig1)
    dst_gaussian2d_7by7_sig1 = my_filtering(src, gaussian2d_mask_7by7_sig1)
    dst_gaussian2d_3by3_sig_dot5 = my_filtering(src, gaussian2d_mask_3by3_sig_dot5)
    dst_gaussian2d_5by5_sig3 = my_filtering(src, gaussian2d_mask_5by5_sig3)
    dst_gaussian2d_7by7_sig_dot1 = my_filtering(src, gaussian2d_mask_7by7_sig_dot1)

    cv2.imwrite('./result/gaussian2D filter 3by3_sig1.jpg', dst_gaussian2d_3by3_sig1)
    cv2.imwrite('./result/gaussian2D filter 5by5_sig1.jpg', dst_gaussian2d_5by5_sig1)
    cv2.imwrite('./result/gaussian2D filter 7by7_sig1.jpg', dst_gaussian2d_7by7_sig1)
    cv2.imwrite('./result/gaussian2D filter 3by3_sig_dot5.jpg', dst_gaussian2d_3by3_sig_dot5)
    cv2.imwrite('./result/gaussian2D filter 5by5_sig3.jpg', dst_gaussian2d_5by5_sig3)
    cv2.imwrite('./result/gaussian2D filter 7by7_sig_dot1.jpg', dst_gaussian2d_7by7_sig_dot1)

    # 1D Gaussian filter
    gaussian1d_mask_3by3_sig1 = my_mask('gaussian1D', 3, sigma=1)
    dst_gaussian1d_3by3_sig1 = my_filtering(src, gaussian1d_mask_3by3_sig1.T)
    dst_gaussian1d_3by3_sig1 = my_filtering(dst_gaussian1d_3by3_sig1, gaussian1d_mask_3by3_sig1)
    cv2.imwrite('./result/gaussian1D filter 3by3_sig1.jpg', dst_gaussian1d_3by3_sig1)

    gaussian1d_mask_5by5_sig1 = my_mask('gaussian1D', 5, sigma=1)
    dst_gaussian1d_5by5_sig1 = my_filtering(src, gaussian1d_mask_5by5_sig1.T)
    dst_gaussian1d_5by5_sig1 = my_filtering(dst_gaussian1d_5by5_sig1, gaussian1d_mask_5by5_sig1)
    cv2.imwrite('./result/gaussian1D filter 5by5_sig1.jpg', dst_gaussian1d_5by5_sig1)

    gaussian1d_mask_7by7_sig1 = my_mask('gaussian1D', 7, sigma=1)
    dst_gaussian1d_7by7_sig1 = my_filtering(src, gaussian1d_mask_7by7_sig1.T)
    dst_gaussian1d_7by7_sig1 = my_filtering(dst_gaussian1d_7by7_sig1, gaussian1d_mask_7by7_sig1)
    cv2.imwrite('./result/gaussian1D filter 7by7_sig1.jpg', dst_gaussian1d_7by7_sig1)

    gaussian1d_mask_3by3_sig_dot5 = my_mask('gaussian1D', 3, sigma=0.5)
    dst_gaussian1d_3by3_sig_dot5 = my_filtering(src, gaussian1d_mask_3by3_sig_dot5.T)
    dst_gaussian1d_3by3_sig_dot5 = my_filtering(dst_gaussian1d_3by3_sig_dot5, gaussian1d_mask_3by3_sig_dot5)
    cv2.imwrite('./result/gaussian1D filter 3by3_sig_dot5.jpg', dst_gaussian1d_3by3_sig_dot5)

    gaussian1d_mask_5by5_sig3 = my_mask('gaussian1D', 5, sigma=3)
    dst_gaussian1d_5by5_sig3 = my_filtering(src, gaussian1d_mask_5by5_sig3.T)
    dst_gaussian1d_5by5_sig3 = my_filtering(dst_gaussian1d_5by5_sig3, gaussian1d_mask_5by5_sig3)
    cv2.imwrite('./result/gaussian1D filter 5by5_sig3.jpg', dst_gaussian1d_5by5_sig3)

    gaussian1d_mask_7by7_sig_dot1 = my_mask('gaussian1D', 5, sigma=0.1)
    dst_gaussian1d_7by7_sig_dot1 = my_filtering(src, gaussian1d_mask_7by7_sig_dot1.T)
    dst_gaussian1d_7by7_sig_dot1 = my_filtering(dst_gaussian1d_7by7_sig_dot1, gaussian1d_mask_7by7_sig_dot1)
    cv2.imwrite('./result/gaussian1D filter 7by7_sig_dot1.jpg', dst_gaussian1d_7by7_sig_dot1)

    cv2.waitKey()
    cv2.destroyAllWindows()
