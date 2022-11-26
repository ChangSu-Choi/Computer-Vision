import numpy as np
import cv2
from tqdm import tqdm

def get_matching_keypoints(img1, img2, keypoint_num):
    '''
    :param img1: 변환시킬 이미지
    :param img2: 변환 목표 이미지
    :param keypoint_num: 추출한 keypoint의 수
    :return: img1의 특징점인 kp1, img2의 특징점인 kp2, 두 특징점의 매칭 결과
    '''
    sift = cv2.xfeatures2d.SIFT_create(keypoint_num)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.DIST_L2)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    """
    matches: List[cv2.DMatch]
    cv2.DMatch의 배열로 구성

    matches[i]는 distance, imgIdx, queryIdx, trainIdx로 구성됨
    trainIdx: 매칭된 img1에 해당하는 index
    queryIdx: 매칭된 img2에 해당하는 index

    kp1[queryIdx]와 kp2[trainIdx]는 매칭된 점
    """
    return kp1, kp2, matches


# bilinear interpolation
def my_bilinear(img, x, y):
    '''
    :param img: 값을 찾을 img
    :param x: interpolation 할 x좌표
    :param y: interpolation 할 y좌표
    :return: img[x,y]에서의 value (bilinear interpolation으로 구해진)
    '''

    floorX, floorY = int(x), int(y)

    t, s = x - floorX, y - floorY

    zz = (1 - t) * (1 - s)
    zo = t * (1 - s)
    oz = (1 - t) * s
    oo = t * s

    interVal = img[floorY, floorX, :] * zz + img[floorY, floorX + 1, :] * zo + \
               img[floorY + 1, floorX, :] * oz + img[floorY + 1, floorX + 1, :] * oo

    # gaussian = GaussianFiltering

    return interVal


def Gbackward(src, target_src, M):
    '''
    :param img1: 변환시킬 이미지
    :param M: 변환 matrix
    :return: 변환된 이미지
    '''
    h, w, c = src.shape
    h_, w_, c_ = target_src.shape

    dst = np.zeros((h_, w_, c_))

    for row in range(h_):
        for col in range(w_):
            xy_prime = np.array([[col, row, 1]]).T
            xy = (np.linalg.inv(M)).dot(xy_prime)

            x_ = xy[0, 0]
            y_ = xy[1, 0]

            if x_ < 0 or y_ < 0 or (x_ + 1) >= w or (y_ + 1) >= h:
                continue

            dst[row, col, :] = my_bilinear(src, x_, y_)

    dst = dst.astype(np.uint8)

    # dst = cv2.GaussianBlur(dst, (5, 5), 0.5) 라이브러리로도 가능
    dst = myGaussian(dst)

    return dst.astype(np.uint8)


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

    for row in range(h):
        for col in range(w):
            val = np.sum(pad_img[row : row + m_h, col : col + m_w] * mask[0 : m_h, 0 : m_w])
            val = np.clip(val, 0, 255) #범위를 0~255로 조정
            dst[row, col] = val

    dst = (dst + 0.5).astype(np.uint8)  # uint8의 형태로 조정

    return dst

def my_zero_padding(src, pad_shape):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h : p_h + h, p_w : p_w + w] = src
    return pad_img

def myGaussian(src):
    h, w, c = src.shape
    mask = my_get_Gaussian2D_mask(msize = (5, 5), sigma = 0.5)
    dst = np.zeros((h, w ,c))

    dst_b, dst_g, dst_r = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    dst_b, dst_g, dst_r = my_filtering(dst_b, mask), my_filtering(dst_g, mask), my_filtering(dst_r, mask)
    dst = cv2.merge((dst_b, dst_g, dst_r))

    return dst

def my_get_Gaussian2D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 2D gaussian filter 만들기
    #########################################
    h, w = msize
    y, x = np.mgrid[-(h // 2):(h // 2) + 1, -(w // 2):(w // 2) + 1]
    # 2차 gaussian mask 생성
    gaus2D = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)

    return gaus2D

if __name__ == '__main__' :
    img1 = cv2.imread('../imgs/Lena.png')
    img1 = cv2.resize(img1, None, fx=0.5, fy=0.5)
    img2 = cv2.imread('../imgs/LenaFaceShear.png')
    kp1, kp2, matches = get_matching_keypoints(img1, img2, keypoint_num = None)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Affine matrix 구하기
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warping (코드로 구현해야함 일단 구현해보자)
    # im_out_real = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

    Gmask = my_get_Gaussian2D_mask((5, 5), 0.5)

    im_out = Gbackward(img1, img2, M)
    im_out_real = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))



    cv2.imshow("Input", img1)
    cv2.imshow("Goal", img2)
    cv2.imshow("Warped Soutce Image", im_out)

    cv2.imshow("Warped Soutce Image-real", im_out_real)


    cv2.waitKey(0)