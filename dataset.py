import pandas as pd
import math
import tensorflow as tf
from PIL import Image
import os
import numpy as np
from config import BATCH_SIZE, CSV_PATH, S,IMAGE_SIZE,ANCHORS,IMAGE_PATH,LABEL_PATH,CSV_PATH
from utils import iou_zeropoint

class YoloDataset(tf.keras.utils.Sequence):
  def __init__(self,csv_file,image_dir,label_dir,anchors,image_size,S,batch_size):
    self.csv_file = pd.read_csv(csv_file)
    self.x = self.csv_file.img.values
    self.y = self.csv_file.label.values
    self.image_dir = image_dir
    self.label_dir = label_dir
    self.anchors = np.array(anchors).reshape(-1,2)
    self.image_size = image_size
    self.S = S
    self.batch_size = batch_size

  def __len__(self):
    return math.ceil(len(self.x)/self.batch_size)
    
  def __getitem__(self,idx):
    batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
    batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]
    batched_x = self.__getX(batch_x)
    batched_y = self.__getY(batch_y)
    for i in range(3):
      batched_y[i] = np.array(batched_y[i])
    return (batched_x,batched_y)
  
  def __getX(self,images):
    batched_image = []
    #loop file
    for filePath in images:
      path = os.path.join(self.image_dir,filePath)
      image = Image.open(path)
      resizedImage = image.resize((self.image_size,self.image_size))
      resizedImage = np.array(resizedImage,dtype=np.float32)
      resizedImage /= 255.
      batched_image.append(resizedImage)
    return np.array(batched_image)
  
  def __getY(self,labels):
    #loop file
    #buat target batch,S,S,3,6 #all size
    batched_target=[[],[],[]]
    for filePath in labels:
      path = os.path.join(self.label_dir,filePath)

      with open(path) as f:
        bboxes = f.read().strip().split('\n')

      anchors = [np.zeros((i,i,3,6)) for i in self.S] #shape target i*i for every grid 3 for anchors 6 for label 0 = isobj, 1,2,3,4 = x,y,w,h,5=objclass
      
      for bbox in bboxes:
        bbox = list(map(float,bbox.split(' '))) #kelas,x,y,w,h
        #find the best iou
        wh = np.array(bbox[-2:]).reshape(-1,2)
        getIou = iou_zeropoint(wh,self.anchors)
        bestIou = tf.argsort(getIou,direction='DESCENDING').numpy()
        has_anchor =[False] * 3 #to make sure that 1 bbox for 1 target 
        for anchor_indices in bestIou:
          whichScale = anchor_indices // 3 # 0 for 13, 1 for 26, 2 for 52 
          whichAnchor = anchor_indices % 3 #every anchor have 3
          whatGrid = self.S[whichScale]
          kelas,x,y,w,h = bbox
          gridi,gridj = int(whatGrid*x) , int(whatGrid*y) #find what grid has responsible
          isAnchorTaken = anchors[whichScale][gridi,gridj,whichAnchor,0]
          if not isAnchorTaken and not has_anchor[whichScale]:
            anchors[whichScale][gridi,gridj,whichAnchor,0] = 1
            normalizeWidth,normalizeHeight = w*whatGrid,h*whatGrid
            normalizeX,normalizeY = (whatGrid*x)-gridi,(whatGrid*y)-gridj
            anchors[whichScale][gridi,gridj,whichAnchor,1] = normalizeX
            anchors[whichScale][gridi,gridj,whichAnchor,2] = normalizeY
            anchors[whichScale][gridi,gridj,whichAnchor,3] = normalizeWidth
            anchors[whichScale][gridi,gridj,whichAnchor,4] = normalizeHeight
            anchors[whichScale][gridi,gridj,whichAnchor,5] = int(kelas)
            has_anchor[whichScale] = True

      for i in range(3):
        batched_target[i].append(anchors[i])

    return batched_target
  
if __name__ == '__main__':
  dataset = YoloDataset(CSV_PATH,IMAGE_PATH,LABEL_PATH,ANCHORS,IMAGE_SIZE,S,BATCH_SIZE)
  assert len(dataset) == 1
  x,y = next(iter(dataset))
  assert x.shape == (1,416,416,3)
  for _ in range(3):
    assert y[_].shape == (1,S[_],S[_],3,6)
  #tes 1 
  assert(y[2][0,33,29,2,:] - np.array([1,0.228,0.51351351,37.336,43.72372372,6]) < np.array([1e-3]*6)).all()
  print('all_test_passed')
    