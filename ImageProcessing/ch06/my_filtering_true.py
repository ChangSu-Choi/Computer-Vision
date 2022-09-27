import cv2
import numpy as np

def my_padding(src, filter):
    (h, w) = src.shape
    (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    pad_img = np.zeros((h+2*h_pad, w+2*w_pad))
    pad_img[h_pad:h+h_pad, w_pad:w+w_pad] = src
    return pad_img

def my_filtering(src, filter):
    (h, w) = src.shape
    #mask의 크기
    (m_h, m_w) = filter.shape
    pad_img = my_padding(src, filter)
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            val = np.sum(pad_img[row:row + m_h, col:col + m_w] * filter)
            dst[row, col] = val

    return dst