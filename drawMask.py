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
output_height = 700
output_width = 400

for i, img in enumerate(imgs):
    print('Original Dimensions : ', img.shape)
    resized = cv2.resize(img, (output_width, output_height), interpolation=cv2.INTER_AREA)
    masked = []

    for r in resized:
        new_c = []
        for c in r:
            # check if the pixel value is green mask or original image pixels
            if True in (abs([120, 255, 155] - c) > [30, 30, 30]):
                new_c.append([0, 0, 0])
            else:
                # keep the green mask pixels in the img
                new_c.append([1, 1, 1])
        masked.append(new_c)
    masked = np.array(masked, dtype=np.uint8)
    if i == 0:
        mask = masked
    else:
        mask = np.ceil((mask + masked) / 2)

    # output the results of each iteration after mask addition process
    #cv2.imwrite('dataset/Mask/Mask-' + str(i) + '.jpg', 255 - (255 * mask))
    #print('Resized Dimensions : ', resized.shape)

# output final mask addition result
cv2.imwrite('dataset/Mask/Mask.jpg', 255 - (255 * mask))

# shrink the mask area
shrink_percentage = 0.1  # shrink the mask by percentage from 4 orientations (top, bottom, left and right)
shrank_height = output_height * (1 - 2 * shrink_percentage)  # shrink top and bottom
shrank_width = output_width * (1 - 2 * shrink_percentage)  # shrink left and right
shrank_mask = cv2.resize(mask, (int(shrank_width), int(shrank_height)), interpolation=cv2.INTER_AREA)
# extend the border of the shrank_mask
shrank_mask = cv2.copyMakeBorder(
    shrank_mask,
    top=int(output_height * shrink_percentage),
    bottom=int(output_height * shrink_percentage),
    left=int(output_width * shrink_percentage),
    right=int(output_width * shrink_percentage),
    borderType=cv2.BORDER_CONSTANT,
    value=[255, 255, 255]
)
cv2.imwrite('dataset/Mask/Mask_shrink.jpg', 255 - (255 * shrank_mask))
