import numpy as np
import os
import matplotlib.image as mpimg
from PIL import Image
from helpers import *
from constants import *
TEST_SIZE = 50

# Get prediction for given input image
#https://github.com/yannvon/road-segmentation/tree/master
def create_submission(model, window_size):
    submission_filename = 'submission__epochs.csv'
    prediction_test_dir = "predictions__epochs/"
    pred_filenames = []
    for i in range(TEST_SIZE):
        image_filename = '/content/drive/MyDrive/Colab Notebooks/datasets/test_set_images/test_' + str(i+1) +"/test_"+ str(i+1) +".png"
        test_imgs = mpimg.imread(image_filename)  
        pimg = get_prediction(test_imgs,model,window_size)
        #save prediction next to the image
        print(pimg)
        w = pimg.shape[0]
        h = pimg.shape[1]
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(pimg)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        pred_filename = prediction_test_dir + "prediction_" + str(i+1) + ".png"
        Image.fromarray(gt_img_3c).save(pred_filename)
        pred_filenames.append(pred_filename)
    masks_to_submission(submission_filename, *pred_filenames)
