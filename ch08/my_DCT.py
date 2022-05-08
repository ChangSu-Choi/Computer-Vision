import cv2
import numpy as np
#jpeg는 보통 block size = 8
def C(w, n = 8):
    if w == 0:
        return (1 / n) ** 0.5
    else:
        return (2 / n) ** 0.5


def Spatial2Frequency_mask(block, n = 8):
    dst = np.zeros(block.shape)
    v, u = dst.shape

    y, x = np.mgrid[0:n, 0:n]

    mask = np.zeros((n*n, n*n))

    for v_ in range(v):
        for u_ in range(u):
            ##########################################################################
            # ToDo                                                                   #
            # mask 만들기                                                             #
            # mask.shape = (16x16)                                                   #
            # DCT에서 사용된 mask는 (4x4) mask가 16개 있음 (u, v) 별로 1개씩 있음 u=4, v=4  #
            ##########################################################################
            tmp = block * np.cos(((2*x+1)*u_*np.pi)/(2*n)) * np.cos(((2*y+1)*v_*np.pi)/(2*n))
            tmp = C(u_, n=n) * C(v_, n=n) * tmp
            mask[4*v_:4*(v_+1), 4*u_:4*(u_+1)] = my_normalize(tmp)
    return mask

def my_normalize(src):
    # 출력 결과를 0 ~ 255로 표현하기 위한 함수
    dst = src.copy()
    if dst.min() != dst.max():
        dst = dst - dst.min()
    dst = dst / dst.max()
    dst = dst * 255
    return dst

if __name__ == '__main__':
    block_size = 4
    src = np.ones((block_size, block_size))

    mask = Spatial2Frequency_mask(src, n=block_size)
    mask = mask.astype(np.uint8)
    print(mask)

    #크기가 너무 작으니 크기 키우기 (16x16) -> (320x320)
    mask = cv2.resize(mask, (320, 320), interpolation=cv2.INTER_NEAREST)

    cv2.imshow('mask', mask)
    cv2.imwrite('./result/mask.png', mask)
    cv2.waitKey()
    cv2.destroyAllWindows()