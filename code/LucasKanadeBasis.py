import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases):
    """
    # Input:
    #	It: template image
    #	It1: Current image
    #	rect: Current position of the car
    #	(top left, bot right coordinates)
    #	bases: [n, m, k] where nxm is the size of the template.
    # Output:
    #	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    """

    p_init = np.zeros(2)

    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]

    rect_len_x = x2 - x1 + 1
    rect_len_y = y2 - y1 + 1

    # Gradients of the current image
    Iy, Ix = np.gradient(It1)

    # Initialize delta_p
    delta_p = np.array([[1], [1]])

    # Threshold
    thresh = 0.05

    b = np.copy(bases)
    b_sum = 0
    b_ortho = b.reshape(-1, bases.shape[2])
    for i in range(bases.shape[2]):
        b_sum += b_ortho[:, i] @ b_ortho[:, i].T

    while np.sqrt(delta_p[0] ** 2 + delta_p[1] ** 2) > thresh:
        px = p_init[0]
        py = p_init[1]

        x1_warped = x1 + px
        y1_warped = y1 + py
        x2_warped = x2 + px
        y2_warped = y2 + py

        # Mesh for full image
        X = np.arange(0, It.shape[0], 1)
        Y = np.arange(0, It.shape[1], 1)
        XX, YY = np.meshgrid(X, Y)
        flat_XX = np.ndarray.flatten(XX)
        flat_YY = np.ndarray.flatten(YY)

        # Mesh for original rectangle
        x = np.linspace(x1, x2, rect_len_x)
        y = np.linspace(y1, y2, rect_len_y)
        xx, yy = np.meshgrid(x, y)
        flat_xx = np.ndarray.flatten(xx)
        flat_yy = np.ndarray.flatten(yy)

        # Mesh for warped rectangle
        x_w = np.linspace(x1_warped, x2_warped, rect_len_x)
        y_w = np.linspace(y1_warped, y2_warped, rect_len_y)
        xx_w, yy_w = np.meshgrid(x_w, y_w)
        flat_xx_w = np.ndarray.flatten(xx_w)
        flat_yy_w = np.ndarray.flatten(yy_w)

        spline_template = RectBivariateSpline(X, Y, It)
        values_template = spline_template.ev(flat_yy, flat_xx)

        spline_current = RectBivariateSpline(X, Y, It1)
        values_warped = spline_current.ev(flat_yy_w, flat_xx_w)

        error = (1 - b_sum) * (values_template - values_warped)

        spline_gradient_x = RectBivariateSpline(X, Y, Ix)
        grad_x_w = spline_gradient_x.ev(flat_yy_w, flat_xx_w)
        spline_gradient_y = RectBivariateSpline(X, Y, Iy)
        grad_y_w = spline_gradient_y.ev(flat_yy_w, flat_xx_w)

        grad_I = np.stack((grad_x_w, grad_y_w), axis=-1)

        # The Jacobian is an identity matrix, since there is only translation in our case
        jacobian = np.array([[1, 0], [0, 1]])

        d = grad_I @ jacobian
        d = (1 - b_sum) * d
        hessian = d.T @ d
        delta_p = np.linalg.inv(hessian) @ d.T @ error

        p_init[0] = p_init[0] + delta_p[0]
        p_init[1] = p_init[1] + delta_p[1]


    p = p_init
    return p

