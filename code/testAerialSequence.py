import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeAffine import LucasKanadeAffine
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':
    vid = np.load("../data/aerialseq.npy")
    masks = np.ones((vid[:, :, 0].shape[0], vid[:, :, 0].shape[1], 4), dtype=bool)
    for f in range(vid.shape[2] - 1):
        img1 = vid[:, :, f]
        img2 = vid[:, :, f + 1]
        mask = SubtractDominantMotion(img1, img2)
        cnt = 0
        if f == 31 or f == 61 or f == 91 or f == 121:
            masks[:, :, cnt] = mask
            cnt += 1

            disp = img2.copy()
            disp = np.stack((disp, disp, disp), axis=2) * 255.0
            disp[:, :, 2] += (mask.astype(np.float32)) * 100.0

            visual = np.clip(disp, 0, 255).astype(np.uint8)
            fig = plt.figure()
            plt.imshow(visual)

            plt.show()
            plt.close()

    print(masks.shape)
    np.save("../outputs/aerialseqmasks.npy", masks)