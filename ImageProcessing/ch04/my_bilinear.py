import cv2
import numpy as np


def my_bilinear(src, scale):
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst))

    # bilinear interpolation 적용
    for row in range(h_dst):
        for col in range(w_dst):
            y = row / scale
            x = col / scale

            m = int(y)
            n = int(x)

            t = -(m-y)
            s = -(n-x)

            """
            픽셀 위치가 이미지를 넘어서는 경우
            1. m+1, n+1 모두 이미지를 넘어서는 경우
            2. m+1이 이미지를 넘어서는 경우 
            3. n+1이 이미지를 넘어서는 경우
            4. 그외
            """
            if m+1 > h-1  and n+1 > w-1:
                value = (1-s)*(1-t)*src[m][n]+s*(1-t)*src[m][n]+(1-s)*t*src[m][n]+s*t*src[m][n]
            elif m+1 > h-1:
                value = (1-s)*(1-t)*src[m][n]+s*(1-t)*src[m][n+1]+(1-s)*t*src[m][n]+s*t*src[m][n+1]
            elif n+1 > w-1:
                value = (1-s)*(1-t)*src[m][n]+s*(1-t)*src[m][n]+(1-s)*t*src[m+1][n]+s*t*src[m+1][n]
            else:
                value = (1-s)*(1-t)*src[m][n]+s*(1-t)*src[m][n+1]+(1-s)*t*src[m+1][n]+s*t*src[m+1][n+1]


            dst[row, col] = value

    # print(count)
    return dst


# Nearest-neighbor interpolation
def my_nearest_neighbor(src, scale):
    (h, w) = src.shape
    h_dst = int(h*scale+0.5)
    w_dst = int(h*scale+0.5)

    dst = np.zeros((h_dst, w_dst), np.uint8)
    for row in range(h_dst):
        for col in range(w_dst):
            r = min(int(row/scale + 0.5), h-1)
            c = min(int(col/scale + 0.5), w-1)
            dst[row, col] = src[r, c]

    return dst


if __name__ == '__main__':
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)

    scale = 5
    #이미지 크기 ??x??로 변경
    my_dst_mini_nearest = my_nearest_neighbor(src, 1/scale)
    my_dst_mini_nearest = my_dst_mini_nearest.astype(np.uint8)

    #이미지 크기 512x512로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst_nearest = my_nearest_neighbor(my_dst_mini_nearest, scale)
    my_dst_nearest = my_dst_nearest.astype(np.uint8)



    #이미지 크기 ??x??로 변경
    my_dst_mini_bilinear = my_bilinear(src, 1/scale)
    my_dst_mini_bilinear = my_dst_mini_bilinear.astype(np.uint8)

    #이미지 크기 512x512로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst_bilinear = my_bilinear(my_dst_mini_bilinear, scale)
    my_dst_bilinear = my_dst_bilinear.astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini_bilinear)
    cv2.imshow('my bilinear', my_dst_bilinear)


    cv2.imshow('my nearest neighbor', my_dst_mini_nearest)
    cv2.imshow('my nearest', my_dst_nearest)

    cv2.waitKey()
    cv2.destroyAllWindows()