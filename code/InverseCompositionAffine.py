import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1):
    """
        Input:
            It: template image
            It1: Current image
        Output:
            M: the Affine warp matrix [2x3 numpy array]
        put your implementation here
        """
    # Put your implementation here

    p_init = np.zeros(6)

    # Gradients of the current image
    Iy, Ix = np.gradient(It1)

    # Initialize delta_p
    delta_p = np.array([[1], [1], [1], [1], [1], [1]])

    # Threshold
    thresh = 1

    # Instead of computing the gradient, Jacobian and Hessian every iteration
    # We can pre-compute them
    flat_Ix = np.ndarray.flatten(Ix)
    flat_Iy = np.ndarray.flatten(Iy)
    grad_I = np.stack((flat_Ix, flat_Iy), axis=-1)
    d = np.zeros((It.shape[0] * It.shape[1], 6))
    for i in range(It.shape[0]):
        for j in range(It.shape[1]):
            jacobian = np.array([[j, 0, i, 0, 1, 0], [0, j, 0, i, 0, 1]])
            flat_ind = i * It.shape[1] + j
            d[flat_ind] = grad_I[flat_ind] @ jacobian

    # hessian will be 6 x 6
    hessian = d.T @ d

    # Initialize M
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    while np.square(delta_p).sum() > thresh:
        # The warp is not translation only anymore, delta_p and p has 6 values
        p1, p2, p3, p4, p5, p6 = p_init[0], p_init[1], p_init[2], p_init[3], p_init[4], p_init[5]
        # Transformation matrix in homogeneous form
        # M = np.array([[1 + p1, p2, p3], [p4, 1 + p5, p6], [0, 0, 1]])

        # Mesh for full image
        X = np.arange(0, It.shape[0], 1)
        Y = np.arange(0, It.shape[1], 1)
        XX, YY = np.meshgrid(X, Y)
        flat_XX = np.ndarray.flatten(XX)
        flat_YY = np.ndarray.flatten(YY)

        # Coordinates stacked into homogeneous matrix
        coord_homo = np.stack((flat_XX, flat_YY, np.ones(flat_XX.shape[0])), axis=-1)
        # the coordinates of the whole image after warp
        coord_warped = coord_homo @ M

        # Original coordinates mapped to template image values
        spline_template = RectBivariateSpline(X, Y, It)
        values_template = spline_template.ev(flat_YY, flat_XX)

        # Warped coordinates mapped to current image values
        spline_current = RectBivariateSpline(X, Y, It1)
        values_warped = spline_current.ev(coord_warped[:, 0], coord_warped[:, 1])

        # error of the values
        error = values_template - values_warped

        delta_p = np.linalg.inv(hessian) @ d.T @ error
        p1, p2, p3, p4, p5, p6 = p_init[0], p_init[1], p_init[2], p_init[3], p_init[4], p_init[5]
        delta_M = np.array([[1.0 + p1, p2, p3], [p4, 1.0 + p5, p6], [0.0, 0.0, 1.0]])
        M = M @ np.linalg.inv(delta_M)

    return M
