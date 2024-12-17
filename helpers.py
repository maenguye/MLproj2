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
      files = '/content/drive/MyDrive/Colab Notebooks/datasets/training/images/satImage_'
      n = 100
      print("Loading " + str(n) + " images")
      imgs = [mpimg.imread(files + '%.3d' % i + '.png') for i in range(1,n)]
      print(imgs[0][2])

      gt_dir ='/content/drive/MyDrive/Colab Notebooks/datasets/training/groundtruth/satImage_'
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



    submission_filename = 'submission__epochs.csv'
    image_filenames = []
    prediction_test_dir = "predictions__epochs/"
    if not os.path.isdir(prediction_test_dir):
        os.mkdir(prediction_test_dir)
    pred_filenames = []
    for i in range(1, TEST_SIZE+1):
        image_filename = '/content/drive/MyDrive/Colab Notebooks/datasets/test_set_images/test_' + str(i) +"/test_"+ str(i) +".png"
        image_filenames.append(image_filename)
    test_imgs = [mpimg.imread(image_filenames[i]) for i in range(TEST_SIZE)]
    for i in range(TEST_SIZE):
        pimg = get_prediction(test_imgs[i],model,window_size)
        #save prediction next to the image
        cimg = concatenate_images(test_imgs[i], pimg)
        Image.fromarray(cimg).save(prediction_test_dir + "prediction_mask_" + str(i) + ".png")
        w = pimg.shape[0]
        h = pimg.shape[1]
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(pimg)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        pred_filename = prediction_test_dir + "prediction_" + str(i+1) + ".png"
        Image.fromarray(gt_img_3c).save(pred_filename)
        pred_filenames.append(pred_filename)
    masks_to_submission(submission_filename, *pred_filenames)


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



#PERSONAL FUNCTION BUT MAKE IT BETTER
def createSubmission(model, window_size):
    submission_filename = 'submission__epochs.csv'
    image_filenames = []
    prediction_test_dir = "predictions__epochs/"
    if not os.path.isdir(prediction_test_dir):
        os.mkdir(prediction_test_dir)
    pred_filenames = []
    for i in range(1, TEST_SIZE+1):
        image_filename = '/content/drive/MyDrive/Colab Notebooks/datasets/test_set_images/test_' + str(i) +"/test_"+ str(i) +".png"
        image_filenames.append(image_filename)
    test_imgs = [mpimg.imread(image_filenames[i]) for i in range(TEST_SIZE)]
    for i in range(TEST_SIZE):
        pimg = get_prediction(test_imgs[i],model,window_size)
        #save prediction next to the image
        cimg = concatenate_images(test_imgs[i], pimg)
        Image.fromarray(cimg).save(prediction_test_dir + "prediction_mask_" + str(i) + ".png")
        w = pimg.shape[0]
        h = pimg.shape[1]
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(pimg)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        pred_filename = prediction_test_dir + "prediction_" + str(i+1) + ".png"
        Image.fromarray(gt_img_3c).save(pred_filename)
        pred_filenames.append(pred_filename)
    masks_to_submission(submission_filename, *pred_filenames)

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

def pad_image_old(img, padSize):
    is_2d = len(img.shape) < 3
    if is_2d:
        return np.lib.pad(img,((padSize,padSize),(padSize,padSize)),'reflect')
    else:
        return np.lib.pad(img,((padSize,padSize),(padSize,padSize),(0,0)),'reflect')

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

def image_generator_old(images, ground_truths, window_size, batch_size=64, upsample=False, class_weights={}):
    np.random.seed(0)
    imgWidth = images[0].shape[0]
    imgHeight = images[0].shape[1]
    half_patch = IMG_PATCH_SIZE // 2

    padSize = (window_size - IMG_PATCH_SIZE) // 2
    paddedImages = []
    for image in images:
        paddedImages.append(pad_image(image, padSize))

    while True:
        batch_input = []
        batch_output = []
        sample_weights = []  # Initialize list to store sample weights

        # Rotates the whole batch for better performance
        randomIndex = np.random.randint(0, len(images))
        img = paddedImages[randomIndex]
        gt = ground_truths[randomIndex]

        # Rotate with probability 10 / 100
        random_rotation = 0
        if (np.random.randint(0, 100) < 10):
            rotations = [90, 180, 270, 45, 135, 225, 315]
            random_rotation = np.random.randint(0, 7)
            img = rotation(img, imgWidth + 2 * padSize, imgHeight + 2 * padSize, rotations[random_rotation], (imgWidth, imgHeight))
            gt = rotation(gt, imgWidth, imgHeight, rotations[random_rotation], (imgWidth, imgHeight))

        background_count = 0
        road_count = 0
        while len(batch_input) < batch_size:
            x = np.empty((window_size, window_size, 3))
            y = np.empty((window_size, window_size, 3))

            # We need to limit possible centers to avoid having a window in an interpolated part of the image
            # We limit ourselves to a square of width 1/sqrt(2) smaller
            if (random_rotation > 2):
                boundary = int((imgWidth - imgWidth / np.sqrt(2)) / 2)
            else:
                boundary = 0
            center_x = np.random.randint(half_patch + boundary, imgWidth - half_patch - boundary)
            center_y = np.random.randint(half_patch + boundary, imgHeight - half_patch - boundary)

            x = img[center_x - half_patch:center_x + half_patch + 2 * padSize,
                    center_y - half_patch:center_y + half_patch + 2 * padSize]
            y = gt[center_x - half_patch:center_x + half_patch,
                    center_y - half_patch:center_y + half_patch]

            # Vertical flip
            if (np.random.randint(0, 2)):
                x = np.flipud(x)

            # Horizontal flip
            if (np.random.randint(0, 2)):
                x = np.fliplr(x)

            label = value_to_class(np.mean(y))

            # Determine sample weight based on class
            sample_weights = []
            label_index = np.argmax(label)  # Get the index of the class (0 or 1)
            weight = class_weights.get(label_index, 1.0)  # Get the weight for the class
            sample_weights.append(weight) # Append the weight to the list

            # Makes sure we have an even distribution of road and non-road if we oversample
            if not upsample:
                batch_input.append(x)
                batch_output.append(label)
            elif label == [1., 0.]:
                # Case background
                background_count += 1
                if background_count != batch_size // 2:
                    batch_input.append(x)
                    batch_output.append(label)
            elif label == [0., 1.]:
                # Case road
                road_count += 1
                if road_count != batch_size // 2:
                    batch_input.append(x)
                    batch_output.append(label)

        batch_x = np.array( batch_input )
        batch_y = np.array( batch_output )

        yield( batch_x, batch_y )

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

    img_width, img_height = images[0].shape[:2]
    half_patch = window_size // 2

    # Padding size to ensure proper patch extraction near edges
    pad_size = (window_size - IMG_PATCH_SIZE) // 2
    padded_images = [pad_image(image, pad_size) for image in images]

    while True:
        batch_input, batch_output, sample_weights = [], [], []

        # Randomly select an image and its ground truth
        random_idx = np.random.randint(0, len(images))
        img, gt = padded_images[random_idx], ground_truths[random_idx]

        # Random rotation with a 10% probability
        if np.random.rand() < 0.1:
            rotations = [90, 180, 270, 45, 135, 225, 315]
            rotation_angle = np.random.choice(rotations)
            img = rotation(img, img_width + 2 * pad_size, img_height + 2 * pad_size, rotation_angle, (img_width, img_height))
            gt = rotation(gt, img_width, img_height, rotation_angle, (img_width, img_height))

        # Balance background and road samples if upsampling
        background_count, road_count = 0, 0

        while len(batch_input) < batch_size:
            # Limit centers to avoid interpolated parts of the image
            boundary = int((img_width - img_width / np.sqrt(2)) / 2) if np.random.rand() < 0.1 else 0
            center_x = np.random.randint(half_patch + boundary, img_width - half_patch - boundary)
            center_y = np.random.randint(half_patch + boundary, img_height - half_patch - boundary)

            # Extract patches
            x_patch = img[center_x - half_patch:center_x + half_patch + 2 * pad_size,
                          center_y - half_patch:center_y + half_patch + 2 * pad_size]
            y_patch = gt[center_x - half_patch:center_x + half_patch,
                          center_y - half_patch:center_y + half_patch]

            # Random flips
            if np.random.rand() < 0.5:
                x_patch = np.flipud(x_patch)
            if np.random.rand() < 0.5:
                x_patch = np.fliplr(x_patch)

            # Label and weighting
            label = value_to_class(np.mean(y_patch))
            label_index = np.argmax(label)
            weight = class_weights.get(label_index, 1.0)

            if not upsample:
                batch_input.append(x)
                batch_output.append(label)
            elif label == [1., 0.]:
                # Case background
                background_count += 1
                if background_count != batch_size // 2:
                    batch_input.append(x)
                    batch_output.append(label)
            elif label == [0., 1.]:
                # Case road
                road_count += 1
                if road_count != batch_size // 2:
                    batch_input.append(x)
                    batch_output.append(label)
            

            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)
            # Append if conditions are met
            #if not upsample or (label == [1., 0.] and background_count < batch_size // 2) or (label == [0., 1.] and road_count < batch_size // 2):
            #    batch_input.append(x_patch)
            #    batch_output.append(label)
            #    sample_weights.append(weight)

                # Update class counters
            #    if label == [1., 0.]:
            #        background_count += 1
             #   elif label == [0., 1.]:
             #       road_count += 1

        yield (batch_x, batch_y)



