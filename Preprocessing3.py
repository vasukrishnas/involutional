import cv2
from itertools import combinations_with_replacement
from collections import defaultdict

import numpy as np
from numpy.linalg import inv
import cv2

R, G, B = 0, 1, 2  # index for convenience

# Trilateral Guided Filter
def boxfilter(I, r):
    M, N = I.shape
    dest = np.zeros((M, N))
    #print(I)
    
    # cumulative sum over Y axis (tate-houkou no wa)
    sumY = np.cumsum(I, axis=0)
    #print('sumY:{}'.format(sumY))
    # difference over Y axis
    dest[:r + 1] = sumY[r:2*r + 1] # top r+1 lines
    dest[r + 1:M - r] = sumY[2*r + 1:] - sumY[:M - 2*r - 1]
    #print(sumY[2*r + 1:]) # from 2*r+1 to end lines
    #print(sumY[:M - 2*r - 1]) # same lines of above, from start
    #tile replicate sumY[-1] and line them up to match the shape of (r, 1)
    dest[-r:] = np.tile(sumY[-1], (r, 1)) - sumY[M - 2*r - 1:M - r - 1] # bottom r lines

    # cumulative sum over X axis
    sumX = np.cumsum(dest, axis=1)
    #print('sumX:{}'.format(sumX))
    # difference over X axis
    dest[:, :r + 1] = sumX[:, r:2*r + 1] # left r+1 columns
    dest[:, r + 1:N - r] = sumX[:, 2*r + 1:] - sumX[:, :N - 2*r - 1]
    dest[:, -r:] = np.tile(sumX[:, -1][:, None], (1, r)) - sumX[:, N - 2*r - 1:N - r - 1] # right r columns

    #print(dest)

    return dest

def guided(p,I):
    r=5; eps=275
    M, N = 224, 224#p.shape
    base = boxfilter(np.ones((M, N)), r) # this is needed for regularization
    
    # each channel of I filtered with the mean filter. this is myu.
    means = [boxfilter(I[:, :, i], r) / base for i in range(3)]
    
    # p filtered with the mean filter
    mean_p = boxfilter(p, r) / base
    
    # filter I with p then filter it with the mean filter
    means_IP = [boxfilter(I[:, :, i]*p, r) / base for i in range(3)]
    
    # covariance of (I, p) in each local patch
    covIP = [means_IP[i] - means[i]*mean_p for i in range(3)]
    
    # variance of I in each local patch: the matrix Sigma in ECCV10 eq.14
    var = defaultdict(dict)
    for i, j in combinations_with_replacement(range(3), 2):
        var[i][j] = boxfilter(I[:, :, i]*I[:, :, j], r) / base - means[i]*means[j]
    
    a = np.zeros((M, N, 3))
    for y, x in np.ndindex(M, N):
        #         rr, rg, rb
        # Sigma = rg, gg, gb
        #         rb, gb, bb
        Sigma = np.array([[var[R][R][y, x], var[R][G][y, x], var[R][B][y, x]],
                          [var[R][G][y, x], var[G][G][y, x], var[G][B][y, x]],
                          [var[R][B][y, x], var[G][B][y, x], var[B][B][y, x]]])
        cov = np.array([c[y, x] for c in covIP])
        a[y, x] = np.dot(cov, inv(Sigma + eps*np.eye(3)))  # eq 14
    
    # ECCV10 eq.15
    b = mean_p - a[:, :, R]*means[R] - a[:, :, G]*means[G] - a[:, :, B]*means[B]
    
    # ECCV10 eq.16
    q = (boxfilter(a[:, :, R], r)*I[:, :, R] + boxfilter(a[:, :, G], r)*I[:, :, G] + boxfilter(a[:, :, B], r)*I[:, :, B] + boxfilter(b, r)) / base
    
    q = q.astype('uint8')
    return q



def CTri_LGF(img):
    img = cv2.normalize(img, None, alpha=0,beta=243, norm_type=cv2.NORM_MINMAX)
    r,g,b = cv2.split(img)
    im = guided(r,img)
    im1 = guided(g,img)
    im2 = guided(b,img)
    GuidedImg = cv2.merge([im, im1, im2])
    GuidedImg = cv2.fastNlMeansDenoising(img, None, 20, 7, 21) 
    return GuidedImg


