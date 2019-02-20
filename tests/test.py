import PyCudaSampling as pcs
import numpy as np
import scipy.misc
import cv2
from timeit import default_timer as timer

# im = scipy.misc.imread('lena.png')
img0 = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# GPU warning up
data_pcs = pcs.sampling(img0.astype(np.float32), 3, [[1,1]], [0.0], [0.0], [False])

for TIMES in [1, 10, 100, 1000, 10000, 100000]:
    pos = [[256,256]]*TIMES
    theta = [90.0]*TIMES
    scale = [2.0]*TIMES
    flip = [False]*TIMES
    PATCH_SIZE = 16
    
    start = timer()
    
    data_opencv = np.empty((TIMES, PATCH_SIZE*PATCH_SIZE))
    for i in range(TIMES):
        rows0, cols0 = img0.shape
        M = np.float32([[1,0,0],[0,1,0]])
        img1 = cv2.warpAffine(img0,M,(cols0,rows0))    
        img2 = cv2.resize(img1, (int(round(scale[i]*cols0)), int(round(scale[i]*rows0))), interpolation = cv2.INTER_CUBIC)
        rows2, cols2 = img2.shape
        M = cv2.getRotationMatrix2D((scale[i]*pos[i][0], scale[i]*pos[i][1]), theta[i], 1)
        img3 = cv2.warpAffine(img2,M,(cols2,rows2))
        
        py0 = int(round(scale[i]*pos[i][1])-PATCH_SIZE/2)
        py1 = int(round(scale[i]*pos[i][1])+PATCH_SIZE/2)
        px0 = int(round(scale[i]*pos[i][0])-PATCH_SIZE/2)
        px1 = int(round(scale[i]*pos[i][0])+PATCH_SIZE/2)
        data_opencv[i] = img3[py0:py1, px0:px1].reshape(PATCH_SIZE*PATCH_SIZE)
    #         scipy.misc.imsave('result_opencv.png', img3[py0:py1, px0:px1].reshape(PATCH_SIZE, PATCH_SIZE))
            
    # ...
    end = timer()
    opencv_time = end - start;
    
    start = timer()
    data_pcs = pcs.sampling(
        img0.astype(np.float32), 
        PATCH_SIZE, 
        pos, # np.array(pos).astype(np.int32).tolist(), 
        theta, # np.array(theta).astype(np.float32).tolist(), 
        scale, # np.array(scale).astype(np.float32).tolist(), 
        flip, # np.array(flip).astype(np.bool).tolist()
    )
    
    end = timer()
    pcs_time = end - start;
    
    print("TIMES:", TIMES, "OpenCV:", opencv_time, "PCS:", pcs_time)

# scipy.misc.imsave('result_pcs.png', data[0].reshape(PATCH_SIZE, PATCH_SIZE))
