import cv2
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


path = 'dataset/Experiment_5.1/test/defect'
mask = cv2.imread(os.path.join('dataset/Mask/', 'Mask-182.jpg'))
mask = mask / 255
imgs = load_images_from_folder(path)

for i, img in enumerate(imgs):
    img = cv2.resize(img, (400, 700), interpolation=cv2.INTER_AREA)
    masked_img = img * mask
    cv2.imwrite('dataset/Experiment_5.5/test/defect/defect-Masked-' + str(i) + '.jpg', masked_img)
