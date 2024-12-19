import numpy as np
import os
import matplotlib.image as mpimg

TEST_SIZE = 50

def createSubmission(model, window_size):
    submission_filename = 'submission__test.csv'
    image_filenames = []
    prediction_test_dir = "predictions__test/"
    if not os.path.isdir(prediction_test_dir):
        os.mkdir(prediction_test_dir)
    pred_filenames = []
    for i in range(1, TEST_SIZE+1):
        image_filename = '/content/drive/MyDrive/Colab Notebooks/datasets/test_set_images/test_' + str(i) +"/test_"+ str(i) +".png"
        image_filenames.append(image_filename)
        #test_imgs = [mpimg.imread(image_filename)]
    test_imgs = [mpimg.imread(image_filenames[i]) for i in range(TEST_SIZE)]
    for i in range(TEST_SIZE):
        pimg = get_prediction(test_imgs[i],model,window_size)
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
