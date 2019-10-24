import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
import scipy.ndimage
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from InverseCompositionAffine import InverseCompositionAffine
import cv2

def SubtractDominantMotion(image1, image2):
    # Input:
    #	Images at time t and t+1
    # Output:
    #	mask: [nxm]
    # put your implementation here
    
    mask = np.ones(image1.shape, dtype=bool)

    M = LucasKanadeAffine(image1, image2)

    # If using inverse composition affine
    # M = InverseCompositionAffine(image1, image2)

    image2_warped = scipy.ndimage.affine_transform(image2, M, output_shape=(image2.shape[0], image2.shape[1]))

    image2_warped = binary_erosion(image2_warped)
    image2_warped = binary_dilation(image2_warped)

    diff = np.abs(image1 - image2_warped)
    thresh = 0.66

    mask = (diff > thresh)

    return mask
