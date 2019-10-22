import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeBasis import LucasKanadeBasis
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':
    vid = np.load("../data/sylvseq.npy")
    bases = np.load("../data/sylvbases.npy")
    fig = plt.figure(1)
    x, y, num_frame = vid.shape

    rect = [101, 61, 155, 107]
    rect_0 = [101, 61, 155, 107]
    rect_no_basis = [101, 61, 155, 107]
    rects = []
    rects.append(rect)

    thresh = 5

    for i in range(num_frame - 1):
        print("frame: ", i)
        frame_temp = vid[:, :, i]
        frame_curr = vid[:, :, i + 1]

        p = LucasKanadeBasis(frame_temp, frame_curr, rect, bases)

        rect[0] = rect[0] + p[0]
        rect[1] = rect[1] + p[1]
        rect[2] = rect[2] + p[0]
        rect[3] = rect[3] + p[1]

        print("basis done")
        rects.append(rect)

        # ============================================================== #
        # Match it with the previous frame template
        p_n = LucasKanade(frame_temp, frame_curr, rect_no_basis, np.zeros(2))

        # Correct the original LK's (comparing with the previous template) gradient descent
        # with the template in the first frame, with the original LK's gradient descent as the initial guess
        rect_star = np.zeros(8)
        rect_star[0:4] = rect_no_basis
        rect_star[4:8] = rect_0
        p_n_star = LucasKanade(vid[:, :, 0], frame_curr, rect_star, p_n)

        # If the second step (comparing with first frame) deviates too much from the
        # first step (comparing with previous frame), it might be a problem and we would not want to
        # update the template in that step

        if np.linalg.norm(p_n_star - p_n) <= thresh:
            # Second step is reasonable, update gradient descent with the first image's template
            rect_no_basis[0] = rect_no_basis[0] + p_n_star[0]
            rect_no_basis[1] = rect_no_basis[1] + p_n_star[1]
            rect_no_basis[2] = rect_no_basis[2] + p_n_star[0]
            rect_no_basis[3] = rect_no_basis[3] + p_n_star[1]
        else:
            # Second step deviates too much from the first step, use original LK's gradient descent
            rect_no_basis[0] = rect_no_basis[0] + p_n[0]
            rect_no_basis[1] = rect_no_basis[1] + p_n[1]
            rect_no_basis[2] = rect_no_basis[2] + p_n[0]
            rect_no_basis[3] = rect_no_basis[3] + p_n[1]


        print("no basis done")

        # For display animation

        # y = patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0] + 1, rect[3] - rect[1] + 1,
        #                       linewidth=2, edgecolor='y', facecolor='none')
        # b = patches.Rectangle((rect_no_basis[0], rect_no_basis[1]), rect_no_basis[2] - rect_no_basis[0] + 1,
        #                       rect_no_basis[3] - rect_no_basis[1] + 1, linewidth=2, edgecolor='b', facecolor='none')
        # plt.imshow(frame_curr, cmap='gray')
        # plt.gca().add_patch(b)
        # plt.gca().add_patch(y)
        # plt.pause(0.001)
        # b.remove()
        # y.remove()


        # For frames 1, 100, 300, 400

        if i == 0 or i == 199 or i == 299 or i == 349 or i == 399:
            y = patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0] + 1, rect[3] - rect[1] + 1,
                              linewidth=2, edgecolor='y', facecolor='none')
            b = patches.Rectangle((rect_no_basis[0], rect_no_basis[1]), rect_no_basis[2] - rect_no_basis[0] + 1,
                              rect_no_basis[3] - rect_no_basis[1] + 1, linewidth=2, edgecolor='b', facecolor='none')
            plt.imshow(frame_curr, cmap='gray')
            plt.gca().add_patch(b)
            plt.gca().add_patch(y)
            plt.show()
            plt.pause(5)
            plt.close()
            b.remove()
            y.remove()

    sylvseqrects = np.array(rects)
    np.save("../outputs/sylvseqrects.npy", sylvseqrects)
