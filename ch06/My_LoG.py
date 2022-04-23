import cv2
import numpy as np

# library add
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_filtering_true import my_filtering

def get_LoG_filter(fsize, sigma=1):
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################
    y, x = np.mgrid[-(fsize//2):(fsize//2)+1, -(fsize//2):(fsize//2)+1]

    LoG = (((x ** 2 + y ** 2) / sigma ** 4) - (2 / sigma ** 2)) *\
        np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))

    LoG = LoG - (LoG.sum()/fsize**2)
    return LoG

def main():
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    src = src / 255
    LoG_filter = get_LoG_filter(fsize=9, sigma=1)

    dst = my_filtering(src, LoG_filter)
    print(dst.max(), dst.min())

    # Nomarilzaiton
    dst = np.abs(dst)
    dst = dst - dst.min()
    dst = dst / dst.max()

    cv2.imshow('dst LoG', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imwrite('./result/dst_LoG.png', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

