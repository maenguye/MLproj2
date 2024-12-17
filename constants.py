NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 25
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 100
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
RECORDING_STEP = 0
IMG_PATCH_SIZE = 16
 # Define the model.
IMG_PATCH_SIZE = 16
NUM_CHANNELS = 3
NUM_LABELS = 2
FOREGROUND_THRESHOLD = 0.25
WINDOW_SIZE = 100