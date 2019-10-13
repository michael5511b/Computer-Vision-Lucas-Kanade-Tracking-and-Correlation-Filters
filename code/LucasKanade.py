import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
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
    
    Iy, Ix = np.gradient(It1)

    px = p0[0]
    py = p0[1]

    while smth < thresh:
    	x1_warped = x1 + px
    	y1_warped = y1 + py
    	x2_warped = x2 + px
    	y2_warped = y2 + py


		X = np.linspace(x1_warped, x2_warped, x2_warped - x1_warped)
		Y = np.linspace(x1, x2, x2 - x1) 	

    	X_w = np.linspace(x1_warped, x2_warped, x2_warped - x1_warped)
    	Y_w = np.linspace(y1_warped, y2_warped, y2_warped - y1_warped)

    	xx, yy = np.meshgrid(X, Y)
    	flat = np.stack((np.flatten(xx), np.flatten(yy)), axis=-1)
    	xx_w, yy_w = np.meshgrid(X, Y)
    	flat_w = np.stack((np.flatten(xx_w), np.flatten(yy_w)), axis=-1)

    	It1(flat_w)



    p = p0
    return p

if __name__ == '__main__':
	LucasKanade()
