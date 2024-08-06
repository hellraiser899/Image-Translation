# import scipy.misc
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
# import imageio
import cv2
from PIL import Image
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_test_data(image_path, fine_size=256):
    img = imread(image_path)
    img = cv2.resize(img, [fine_size*2, fine_size])
    img = img/127.5 - 1
    return img

def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    if not is_testing:
        img_A = cv2.resize(img_A, [load_size*2, load_size])
        img_B = cv2.resize(img_B, [load_size*2, load_size])
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, (load_size-fine_size)*2)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size*2]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size*2]
        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
        
    
    else:
        img_A = cv2.resize(img_A, [fine_size*2, fine_size])
        img_B = cv2.resize(img_B, [fine_size*2, fine_size])

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    img_AB = np.concatenate((img_A, img_B), axis=2)
    return img_AB

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return plt.imread(path, flatten = True).astype(float)
    else:
        return plt.imread(path).astype(float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    # print(images.shape)
    # print(images)

    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    # print("final image")
    # print(img.shape)
    # print(img)
    return img

def imsave(images, size, path):
    i=Image.fromarray((merge(images,size)* 255).astype(np.uint8))
    return i.save(path)
    

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return cv2.resize(
      x[j:j+crop_h, i:i+crop_w], [resize_w, resize_h])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images*0.5 + 0.5)


# cv2.imwrite('a.jpg',inverse_transform(load_test_data('image1.jpg'))*255)
# # cv2.waitKey(0)
# i=Image.fromarray((inverse_transform(load_test_data('image1.jpg'))* 255).astype(np.uint8))
# i.save('b.jpg')