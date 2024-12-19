import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code
import matplotlib.pyplot as plt

import tensorflow.python.platform
import sys
import numpy
import tensorflow as tf
import argparse
import numpy as np

from scipy.ndimage import rotate
import os
import numpy
import matplotlib.image as mpimg
import re
from PIL import Image

from constants import *

#GIVEN FUNCTION
def img_crop(im, w, h): #given function
    '''
    Crop an image into patches of size w x h
    '''
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches

def value_to_class(v): #given function
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]

def extract_data(filename, num_images): #given function
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            # print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            # Normalize
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    img_patches = [
        img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ] # list of list of patches, each list of patches corresponds to one image
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ] # flatten the list of patches

    return numpy.asarray(data)

def cropped_imgs(images, patch_size): #given function
    '''
    Crop the labels into patches of size patch_size x patch_size
    '''
    imgs_patches = [
        img_crop(images[i], patch_size, patch_size) for i in range(len(images))
    ]
    return numpy.asarray(
        [
            imgs_patches[i][j]
            for i in range(len(imgs_patches))
            for j in range(len(imgs_patches[i]))
        ]
    )

def cropped_labels(labels, patch_size): #given function

    labels_patches = [
        img_crop(labels[i], patch_size, patch_size) for i in range(len(labels))
    ]

    data = numpy.asarray(
        [
            labels_patches[i][j]
            for i in range(len(labels_patches))
            for j in range(len(labels_patches[i]))
        ]
    )
    labels = np.asarray(
        [value_to_class(np.mean(data[i])) for i in range(len(data))]
    )

    return labels.astype(numpy.float32)


def extract_labels(filename, num_images): #given function
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            # print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(gt_imgs)
    gt_patches = [
        img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = numpy.asarray(
        [
            gt_patches[i][j]
            for i in range(len(gt_patches))
            for j in range(len(gt_patches[i]))
        ]
    )
    labels = numpy.asarray(
        [value_to_class(numpy.mean(data[i])) for i in range(len(data))]
    )

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


def load_data(image_dir, gt_dir, training_size):
    #   files = '/content/drive/MyDrive/Colab Notebooks/datasets/training/images/satImage_'
      files = image_dir + '/satImage_'
      n = 100
      print("Loading " + str(n) + " images")
      imgs = [mpimg.imread(files + '%.3d' % i + '.png') for i in range(1,n)]
      print(imgs[0][2])

    #   gt_dir ='/content/drive/MyDrive/Colab Notebooks/datasets/training/groundtruth/satImage_'
      gt_dir = gt_dir + '/satImage_'
      print("Loading " + str(n) + " images")
      gt_imgs = [mpimg.imread(gt_dir + '%.3d' % i + '.png') for i in range(1,n)]

      X_train = imgs
      Y_train = gt_imgs
      return X_train, Y_train


def img_float_to_uint8(img): #given function
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg


def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH
    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


# Get prediction for given input image
def get_prediction(img, model, window_size):
    data = numpy.asarray(create_windows(img, window_size))
    output_prediction = model.predict(data)
    img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
    return img_prediction


# Get a concatenation of the prediction and image for given input file
def get_prediction_with_mask(img, model, window_size):
    img_prediction = get_prediction(img, model, window_size)
    cimg = concatenate_images(img, img_prediction)
    return cimg


# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(img, model, window_size):
    img_prediction = get_prediction(img,model, window_size)
    oimg = make_img_overlay(img, img_prediction)
    return oimg


# assign a label to a patch
def patch_to_label(patch):
    df = numpy.mean(patch)
    if df > FOREGROUND_THRESHOLD:
        return 1 #inverted before 0
    else:
        return 0 #inverted before 1

def mask_to_submission_strings(img_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", img_filename).group(0))
    im = mpimg.imread(img_filename)
    patch_size = 16
    for j in range(0, im.shape[1], IMG_PATCH_SIZE):
        for i in range(0, im.shape[0], IMG_PATCH_SIZE):
            patch = im[i:i + IMG_PATCH_SIZE, j:j + IMG_PATCH_SIZE]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))





def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*255
    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels): #given function
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx][0] > 0.5: # FIXME make something cleaner?
                l = 0 #inverted before 1
            else:
                l = 1 #inverted before 0
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels


#Method to take care of the values that are 253 or 254 on the groundtruth images
def round(x):
    if(x < 0.5):
        return 1. #inverted before 0.
    else:
        return 0. #inverted before 1.

def rotation(image, x, y, angle, image_shape):
    theta = np.radians(angle)
    im_rotate = rotate(image, angle, reshape=False)
    h, w = image_shape
    center_x, center_y = (w - 1) / 2, (h - 1) / 2
    x_c, y_c = x - center_x, y - center_y
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_c_rot = cos_theta * x_c - sin_theta * y_c
    y_c_rot = sin_theta * x_c + cos_theta * y_c
    x_rot, y_rot = x_c_rot + center_x, y_c_rot + center_y

    return im_rotate

def create_window_old(im, window_size):
    list_patches = []
    is_2d = len(im.shape) < 3
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    padSize = (window_size - IMG_PATCH_SIZE)//2
    padded = pad_image(im, padSize)
    for i in range(padSize, imgheight + padSize, IMG_PATCH_SIZE):
        for j in range(padSize,imgwidth + padSize, IMG_PATCH_SIZE):
            if is_2d:
                im_patch = padded[j-padSize:j+ IMG_PATCH_SIZE+padSize, i-padSize:i+ IMG_PATCH_SIZE+padSize]
            else:
                im_patch = padded[j-padSize:j+ IMG_PATCH_SIZE+padSize, i-padSize:i+ IMG_PATCH_SIZE+padSize, :]
            list_patches.append(im_patch)
    return list_patches

def create_windows(image, window_size):
    """
    Splits an image into overlapping windows of the specified size.

    Args:
        image (np.array): Input image to split.
        window_size (int): Size of each window.

    Returns:
        list: List of image windows.
    """
    img_height, img_width = image.shape[:2]
    pad_size = (window_size - IMG_PATCH_SIZE) // 2
    padded_image = pad_image(image, pad_size)

    windows = []
    for i in range(pad_size, img_height + pad_size, IMG_PATCH_SIZE):
        for j in range(pad_size, img_width + pad_size, IMG_PATCH_SIZE):
            if image.ndim == 2:
                patch = padded_image[j - pad_size:j + IMG_PATCH_SIZE + pad_size, i - pad_size:i + IMG_PATCH_SIZE + pad_size]
            else:
                patch = padded_image[j - pad_size:j + IMG_PATCH_SIZE + pad_size, i - pad_size:i + IMG_PATCH_SIZE + pad_size, :]
            windows.append(patch)

    return windows

def pad_image(image, pad_size):
    """
    Pads an image with the specified pad size.

    Args:
        image (np.array): Input image to pad.
        pad_size (int): Size of padding.

    Returns:
        np.array: Padded image.
    """
    padding = ((pad_size, pad_size), (pad_size, pad_size))
    if image.ndim == 3:
        padding = padding + ((0, 0),)
    return np.pad(image, padding, mode='reflect')


def rotate_image(image, angle, img_width, img_height):
    return rotation(image, img_width, img_height, angle, (img_width, img_height))


def calculate_boundary(rotation, img_width):
    if rotation > 2:
        return int((img_width - img_width / np.sqrt(2)) / 2)
    return 0

def get_random_center(half_patch_size, boundary, img_width, img_height):
    x_center = np.random.randint(half_patch_size + boundary, img_width - half_patch_size - boundary)
    y_center = np.random.randint(half_patch_size + boundary, img_height - half_patch_size - boundary)
    return x_center, y_center

def extract_patch(image, x_center, y_center, half_patch_size, pad):
    return image[x_center - half_patch_size : x_center + half_patch_size + 2 * pad,
                y_center - half_patch_size : y_center + half_patch_size + 2 * pad]

        
def apply_random_flip(patch):
    if np.random.randint(0, 2):
        patch = np.flipud(patch)
    if np.random.randint(0, 2):
        patch = np.fliplr(patch)
    return patch

def image_generator(images, ground_truths, window_size, batch_size=64, upsample=False, class_weights={}):
    """
    A generator function that yields batches of image patches and their corresponding labels.

    Args:
        images (list): List of input images.
        ground_truths (list): Corresponding ground truth images.
        window_size (int): Size of the image patch to extract.
        batch_size (int): Number of samples per batch.
        upsample (bool): Whether to balance class distribution by oversampling.
        class_weights (dict): Dictionary mapping class indices to their weights.

    Yields:
        tuple: A batch of image patches (batch_x) and their labels (batch_y).
    """
    np.random.seed(0)

    height, width = images[0].shape[:2]
    padding = int((window_size - IMG_PATCH_SIZE) / 2)
    padded_img = []
    
    
    for image in images:
        padded_img.append(pad_image(image, padding))

    while True:
        batch_input = []
        batch_output = []
        sample_weights = []  # Initialize list to store sample weights

        # Rotates the whole batch for better performance
        randomIndex = np.random.randint(0, len(images))
        img = padded_img[randomIndex]
        lab = ground_truths[randomIndex]

        # Rotate with probability 10 / 100
        random_rotation = 0
        rotation_chance = np.random.random()
        should_rotate = rotation_chance < 0.1

        if should_rotate:
            angles = [90, 180, 270, 45, 135, 225, 315]
            chosen_angle = angles[np.random.choice(range(len(angles)))]
            rotated_img = rotate_image(img, chosen_angle, width + 2 * padding, height + 2 * padding)
            rotated_lab = rotate_image(lab, chosen_angle, width, height)

        background_count = 0
        road_count = 0
        while len(batch_input) < batch_size:
            x = np.empty((window_size, window_size, 3))
            y = np.empty((window_size, window_size, 3))
            
            patch_boundary = calculate_boundary(random_rotation, width)
            center_x, center_y = get_random_center(int(IMG_PATCH_SIZE / 2), patch_boundary, width, height)
            x = extract_patch(img, center_x, center_y, int(IMG_PATCH_SIZE / 2), padding)
            y = extract_patch(lab, center_x, center_y,  int(IMG_PATCH_SIZE / 2), 0)
            
            x = apply_random_flip(x)

            label = value_to_class(np.mean(y))

            # Determine sample weight based on class
            sample_weights = []
            sample_weights.append(class_weights.get(np.argmax(label), 1.0)) 

            # Makes sure we have an even distribution of road and non-road if we oversample
            # if not upsample:
            #     batch_input.append(x)
            #     batch_output.append(label)
            if label == [1., 0.]:
                # Case background
                if background_count < batch_size // 2:
                    batch_input.append(x)
                    batch_output.append(label)
                    background_count = background_count + 1
            elif label == [0., 1.]:
                # Case road
                if road_count < batch_size // 2:
                    batch_input.append(x)
                    batch_output.append(label)
                    road_count = road_count + 1

        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield( batch_x, batch_y )
        
        
