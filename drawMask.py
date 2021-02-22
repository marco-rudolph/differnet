import cv2
import os
import numpy as np


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:
            images.append(image)
    return images


imgs = load_images_from_folder('dataset/bgm/')

for i, img in enumerate(imgs):
    print('Original Dimensions : ', img.shape)
    resized = cv2.resize(img, (400, 700), interpolation=cv2.INTER_AREA)
    masked = []

    for r in resized:
        new_c = []
        for c in r:
            # check if the pixel value is green mask or original image pixels
            if True in (abs([120, 255, 155] - c) > [30, 30, 30]):
                new_c.append([0, 0, 0])
            else:
                # keep the green mask pixels in the img
                # todo: increase mask area.
                new_c.append([1, 1, 1])
        masked.append(new_c)
    masked = np.array(masked, dtype=np.uint8)
    if i == 0:
        mask = masked
    else:
        mask = np.ceil((mask + masked) / 2)

    # output the results of each iteration after mask addition process
    cv2.imwrite('dataset/Mask/Mask-' + str(i) + '.jpg', 255 - (255 * mask))
    print('Resized Dimensions : ', resized.shape)

# output final mask addition result
cv2.imwrite('dataset/Mask/Mask.jpg', 255 - (255 * mask))
