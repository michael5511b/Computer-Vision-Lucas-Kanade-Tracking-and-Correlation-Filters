import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
ims = []
if __name__ == '__main__':
    vid = np.load("../data/carseq.npy")
    fig = plt.figure(1)
    x, y, num_frame = vid.shape

    rect = [59, 116, 145, 151]
    rects = []
    rects.append(rect)

    for i in range(num_frame - 1):
        frame_temp = vid[:, :, i]
        frame_curr = vid[:, :, i + 1]
        p = LucasKanade(frame_temp, frame_curr, rect, p0=np.zeros(2))

        rect[0] = rect[0] + p[0]
        rect[1] = rect[1] + p[1]
        rect[2] = rect[2] + p[0]
        rect[3] = rect[3] + p[1]

        rects.append(rect)

        # For display animation
        """
        r = patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0] + 1, rect[3] - rect[1] + 1, linewidth=2, edgecolor='r', facecolor='none')
        plt.imshow(frame_curr, cmap='gray')
        plt.gca().add_patch(r)
        plt.pause(0.001)
        r.remove()
        """

        # For frames 1, 100, 300, 400
        """
        if i == 0 or i == 99 or i == 199 or i ==299 or i == 399:
            r = patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0] + 1, rect[3] - rect[1] + 1, linewidth=2, edgecolor='r', facecolor='none')
            plt.imshow(frame_curr, cmap='gray')
            plt.gca().add_patch(r)
            plt.show()
            plt.pause(5)
            plt.close()
            r.remove()
        """
    carseqrects = np.array(rects)
    np.save("../outputs/carseqrects.npy", carseqrects)
