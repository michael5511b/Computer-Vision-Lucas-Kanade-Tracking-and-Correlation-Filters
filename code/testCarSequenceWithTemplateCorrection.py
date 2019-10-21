import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':
    vid = np.load("../data/carseq.npy")
    fig = plt.figure(1)
    x, y, num_frame = vid.shape

    rect = [59, 116, 145, 151]
    rect_no_correct = [59, 116, 145, 151]
    rects = []
    rects.append(rect)

    thresh = 5
    p = np.zeros(2)

    for i in range(num_frame - 1):
        frame_temp = vid[:, :, i]
        frame_curr = vid[:, :, i + 1]

        print("frame: ", i)


        # Match it with the previous frame template
        p_n = LucasKanade(frame_temp, frame_curr, rect, np.zeros(2))

        # Correct the original LK's (comparing with the previous template) gradient descent
        # with the template in the first frame, with the original LK's gradient descent as the initial guess
        p_n_star = LucasKanade(vid[:, :, 0], frame_curr, rect, p_n)

        # If the second step (comparing with first frame) deviates too much from the
        # first step (comparing with previous frame), it might be a problem and we would not want to
        # update the template in that step

        if np.linalg.norm(p_n_star - p_n) <= thresh:
            # Second step is reasonable, update gradient descent with the first image's template
            rect[0] = rect[0] + p_n_star[0]
            rect[1] = rect[1] + p_n_star[1]
            rect[2] = rect[2] + p_n_star[0]
            rect[3] = rect[3] + p_n_star[1]
        else:
            # Second step deviates too much from the first step, use original LK's gradient descent
            rect[0] = rect[0] + p_n[0]
            rect[1] = rect[1] + p_n[1]
            rect[2] = rect[2] + p_n[0]
            rect[3] = rect[3] + p_n[1]

        rects.append(rect)

        # This is for the method without template correction
        # Compare with new method
        p_no_correct = LucasKanade(frame_temp, frame_curr, rect_no_correct, np.zeros(2))
        rect_no_correct[0] = rect_no_correct[0] + p_no_correct[0]
        rect_no_correct[1] = rect_no_correct[1] + p_no_correct[1]
        rect_no_correct[2] = rect_no_correct[2] + p_no_correct[0]
        rect_no_correct[3] = rect_no_correct[3] + p_no_correct[1]


        # For display animation
        """
        r = patches.Rectangle((rect_no_correct[0], rect_no_correct[1]),
                              rect_no_correct[2] - rect_no_correct[0] + 1, rect_no_correct[3] - rect_no_correct[1] + 1,
                              linewidth=2, edgecolor='r', facecolor='none')
        b = patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0] + 1, rect[3] - rect[1] + 1,
                              linewidth=2, edgecolor='b', facecolor='none')
        plt.imshow(frame_curr, cmap='gray')
        plt.gca().add_patch(r)
        plt.gca().add_patch(b)
        plt.pause(0.001)
        r.remove()
        b.remove()
        """

        # For frames 1, 100, 300, 400

        if i == 0 or i == 99 or i == 199 or i ==299 or i == 399:
            r = patches.Rectangle((rect_no_correct[0], rect_no_correct[1]),
                              rect_no_correct[2] - rect_no_correct[0] + 1, rect_no_correct[3] - rect_no_correct[1] + 1,
                              linewidth=2, edgecolor='r', facecolor='none')
            b = patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0] + 1, rect[3] - rect[1] + 1,
                              linewidth=2, edgecolor='b', facecolor='none')
            plt.imshow(frame_curr, cmap='gray')
            plt.gca().add_patch(r)
            plt.gca().add_patch(b)
            plt.show()
            plt.pause(5)
            plt.close()
            r.remove()
            b.remove()

    carseqrects_wcrt = np.array(rects)
    np.save("../outputs/carseqrects-wcrt.npy", carseqrects_wcrt)