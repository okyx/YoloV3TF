# () tuple ordinaty cnn block
# int  residual block
# str u = upsample 
from tkinter import ANCHOR


config = [
    (32,3,1),
    (64,3,2),
    1,
    (128, 3, 2),
    2,
    (256, 3, 2),
    8,
    (512, 3, 2),
    8,
    (1024, 3, 2),
    4,
    (512,1,1),
    (1024,3,1),
    'S',
    (256,1,1),
    'U',
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128,1,1),
    'U',
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

num_classes=20
S = [13,26,52]
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]
IMAGE_SIZE = 416
BATCH_SIZE = 32
LABEL_PATH ='./assets/labels'
CSV_PATH = './assets/tes.csv'
IMAGE_PATH = './assets//images'
NUM_EPOCHS = 25
CLASS_THRESHOLD = 0.8
NMS_THRESHOLD=0.5