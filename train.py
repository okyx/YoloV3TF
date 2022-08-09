import tensorflow as tf
from dataset import YoloDataset
from model import TFYoloModel
from loss import YOLOLoss
from config import BATCH_SIZE, CSV_PATH, NUM_EPOCHS, S,IMAGE_SIZE,ANCHORS,IMAGE_PATH,LABEL_PATH,CSV_PATH,config,num_classes,NUM_EPOCHS
import numpy as np
from tqdm import tqdm

# @tf.function
def train_step(images,labels):
  with tf.GradientTape() as tape:
    prediction = model(images)
    loss = 0
    for _ in range(3):
      loss += YoloLoss(labels[_],prediction[_],scaled[_])
    print(loss)
  grad = tape.gradient(loss,model.trainable_variables)
  optimizer.apply_gradients(zip(grad,model.trainable_variables))
  return loss

def train_step(images,labels):
  with tf.GradientTape() as tape:
    prediction = model(images)
    loss = 0
    for _ in range(3):
      loss += YoloLoss(labels[_],prediction[_],scaled[_])
  return loss
if __name__ == '__main__':
  Traindataset = YoloDataset(CSV_PATH,IMAGE_PATH,LABEL_PATH,ANCHORS,IMAGE_SIZE,S,BATCH_SIZE)
  Testdataset = YoloDataset(CSV_PATH,IMAGE_PATH,LABEL_PATH,ANCHORS,IMAGE_SIZE,S,BATCH_SIZE)
  model = TFYoloModel(config=config,num_classes=num_classes)
  YoloLoss = YOLOLoss()
  ANCHORS = np.array(ANCHORS)
  S = np.array(S).reshape(3,1,1)
  scaled = S*ANCHORS
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
  for _ in tqdm(range(NUM_EPOCHS)):
    trainloss = 0
    testloss = 0
    for x,y in Traindataset:
      trainloss = train_step(x,y)
    for x,y in Testdataset:
      testloss = train_step(x,y)
    print(trainloss,testloss)
    
  