from dataset import YoloDataset
from model import TFYoloModel
from loss import YOLOLoss
from config import BATCH_SIZE, CSV_PATH, S,IMAGE_SIZE,ANCHORS,IMAGE_PATH,LABEL_PATH,CSV_PATH,config,num_classes
import numpy as np

dataset = YoloDataset(CSV_PATH,IMAGE_PATH,LABEL_PATH,ANCHORS,IMAGE_SIZE,S,BATCH_SIZE)
model = TFYoloModel(config=config,num_classes=num_classes)
loss = YOLOLoss()
ANCHORS = np.array(ANCHORS)
S = np.array(S).reshape(3,1,1)
scaled = S*ANCHORS

for x,y in dataset:
  modelOutput = model(x)
  for _ in range(3):
    loss(y[_],modelOutput[_],scaled[_])
  print('all test passed no error')