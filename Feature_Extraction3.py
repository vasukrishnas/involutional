import cv2 
import numpy as np
import pywt
# laplace Transform

def tranform(image):
    x=image
    coffes=pywt.dwt2(x,'haar')
    high,(ch,cv,cd)=coffes
    high2=np.uint8(high)
    ch2=np.uint8(ch)
    cvi2=np.uint8(cv)
    cd2=np.uint8(cd)
    coffes=high2,(ch2,cvi2,cd2)
    low_low_filter=pywt.idwt2(coffes,'haar')
    DirectbankFilter = cv2.dft(np.float32(low_low_filter), flags=cv2.DFT_COMPLEX_OUTPUT)
    shift = np.fft.fftshift(DirectbankFilter)
    row, col = low_low_filter.shape
    e_img = cv2.Laplacian(image, -1, ksize=5, scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)   
    center_row, center_col = row // 2, col // 2
    mask = np.zeros((row, col, 2), np.uint8)
    mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 1
    fft_shift = shift * mask
    fft_ifft_shift = np.fft.ifftshift(fft_shift)
    imageThen = cv2.idft(fft_ifft_shift)
    e_img_ = cv2.magnitude(imageThen[:,:,0], imageThen[:,:,1])
    return e_img

def laplace_transf(Extraction):
    
    img = Extraction.max(2)
    extract_img = cv2.Laplacian(img, -1, ksize=5, scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)   
    kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
    CT_img = cv2.filter2D(src=extract_img, ddepth=-1, kernel=kernel)
    
    return CT_img