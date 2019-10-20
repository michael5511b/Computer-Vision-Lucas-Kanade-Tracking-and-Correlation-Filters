import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(It, It1, rect, p0=np.zeros(2)):
	# Input:
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	# Put your implementation here
	x1 = rect[0]
	y1 = rect[1]
	x2 = rect[2]
	y2 = rect[3]

	rect_len_x = x2 - x1 + 1
	rect_len_y = y2 - y1 + 1

	Iy, Ix = np.gradient(It1)

	px = p0[0]
	py = p0[1]

	delta_p = np.array([[1], [1]])
	thresh = 0.1


	while np.sqrt(delta_p[0] ** 2 + delta_p[1] ** 2) > thresh:
		px = p0[0]
		py = p0[1]

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



		# Create RectBivariateSpline object
		# given a rectangle, RectBivariateSpline will return an "Object"
		# We can then give the object non-integer indices, and the object would spit out interpolated values

		spline_template = RectBivariateSpline(X, Y, It)
		# .ev is tricky, the x and y are flipped!
		values_template = spline_template.ev(flat_yy, flat_xx)

		# Warping the current image to look like the template
		# thus we use the warped coordinates, look at those values of the warped coordinates at the current image
		# and compare the error to the template image

		# In the Baker_Simon paper:
		# The goal of the Lucas-Kanade algorithm is to minimize the sum of squared error between two
		# images, the template T and the image I warped back onto the "coordinate frame of the template"

		# Σ [I(W(x; p)) - T(x)]^2
		# > I is the current image
		# > T is the template image
		# > W(x; p) take the pixel x in the coordinate frame of the template T and and maps it to the sub-pixel location W(x; p)
		# in the coordinate frame of image I. So W is representing "coordinates"

		spline_current = RectBivariateSpline(X, Y, It1)
		# This right here is I(W(x; p))
		values_warped = spline_current.ev(flat_yy_w, flat_xx_w)

		error = values_template - values_warped

		spline_gradient_x = RectBivariateSpline(X, Y, Ix)
		grad_x_w = spline_gradient_x.ev(flat_yy_w, flat_xx_w)
		spline_gradient_y = RectBivariateSpline(X, Y, Iy)
		grad_y_w = spline_gradient_y.ev(flat_yy_w, flat_xx_w)

		grad_I = np.stack((grad_x_w, grad_y_w), axis=-1)

		# The Jacobian is an identity matrix, since there is only translation in our case
		jacobian = np.array([[1, 0], [0, 1]])

		# The acquisition of delta_p is derived from the First Order Taylor Series
		# linearization of the square error function, check the paper!
		# @ is matrix multiplication, it's the equivalent of matmul()
		hessian = (grad_I @ jacobian).T @ (grad_I @ jacobian)
		delta_p = np.linalg.inv(hessian) @ (grad_I @ jacobian).T @ error

		p0[0] = p0[0] + delta_p[0]
		p0[1] = p0[1] + delta_p[1]

	p = p0
	return p
