# import the necessary packages
import os

# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "E:\\screenshot\\"

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
#BASE_PATH = "/home/hzhuang/Projects/Predatory_Journal/release/screenshot_for_training"
BASE_PATH = "E:\\SU_Backup\\Predatory_Journal\\release\\screenshot_training\\"

# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "test"])

# define the names of the training, testing, and validation
# directories
TRAIN = "training"
TEST = "test"
VAL = "validation"

# define the amount of data that will be used training
TRAIN_SPLIT = 0.8

# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.2

# define the names of the classes
CLASSES = ["in_DOAJ", "out_DOAJ"]

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 32
BATCH_SIZE = 32
NUM_EPOCHS = 10

# define the path to the serialized output model after training
MODEL_PATH = "binary_classification_2.model"

# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "output"