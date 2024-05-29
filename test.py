import os
import cv2
import numpy as np
from scipy import io
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

depth = cv2.imread('RGBD_3_0.EXR', -1)[:,:,3]

depth_min = np.min(depth)

depth -= depth_min

io.savemat('true_depth.mat', {'true_depth': depth})

