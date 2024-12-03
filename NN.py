import pandas as pd
import numpy as np
import os 
import glob
from PIL import Image
import cv2
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from functions import *
import matplotlib.image as mpimg


# Extract patches from a given image
def img_crop(im, w, h):
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

def load_image(image_path):
    # Load PNG image as a NumPy array
    image = cv2.imread(image_path)
    gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    normalized = gray / 255.0
    blur = cv2.GaussianBlur(normalized, (5, 5), 0) #can find the best kernel size 
    #edges = cv2.Canny((blur * 255).astype('uint8'), 100, 200)
    #lines = cv2.HoughLinesP(blur, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    # Convert to uint8 for morphological processing
    processed = (blur * 255).astype('uint8')
    # Apply morphological operations to enhance linear structures
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))  # Define kernel for elongation
    #morph = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)  # Closing to connect structures
    # Perform Hough Line Transform directly on the morphologically processed image
    #lines = cv2.HoughLinesP(morph, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    return processed

def extract_data(filename, num_images,IMG_PATCH_SIZE):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename) #without filters 
            #img = load_image(image_filename) #with filters 
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    img_patches = [
        img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]

    return np.asarray(data)

# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]


# Extract label images
def extract_labels(filename, num_images,IMG_PATCH_SIZE):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(gt_imgs)
    gt_patches = [
        img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = np.asarray(
        [
            gt_patches[i][j]
            for i in range(len(gt_patches))
            for j in range(len(gt_patches[i]))
        ]
    )
    labels = np.asarray(
        [value_to_class(np.mean(data[i])) for i in range(len(data))]
    )

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)
