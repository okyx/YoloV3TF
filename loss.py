import tensorflow as tf
from utils import iou
from config import ANCHORS,S,num_classes
import numpy as np

class YOLOLoss(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    self.mse = tf.keras.losses.MeanSquaredError()
    self.sparse = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.lambda_class = 1
    self.lambda_noobj = 10
    self.lambda_obj = 1
    self.lambda_box = 10
  
  def call(self,target,pred,anchors):
    '''
    target (batch,s,s,3,6)
    pred (batch,s,s,3*(class+5))
    '''
    shape = pred.shape
    prediction = tf.reshape(pred,shape=[shape[0],shape[1],shape[2],3,shape[3]//3])

    isObj = target[...,0] ==1
    isNotObj = target[...,0]==0

    #no obj loss
    noobjloss = self.bce(
        target[...,0:1][isNotObj],prediction[...,0:1][isNotObj]
    )
    #class loss
    classloss = self.sparse(
        target[...,5][isObj],
        prediction[...,5:][isObj]
      )
    
    #box loss
    anchors = tf.reshape(anchors,shape=(1,1,1,3,2))
    target = target.numpy()
    prediction = prediction.numpy()
    prediction[...,1:3] = tf.keras.activations.sigmoid(prediction[...,1:3])
    target[...,3:5] = tf.math.log (
        1e-16 + target[...,3:5]/anchors
    )
    target = tf.convert_to_tensor(target)
    prediction = tf.convert_to_tensor(prediction)
    boxloss = self.mse(target[...,1:5][isObj],prediction[...,1:5][isObj])

    #obj loss
    box_pred = tf.concat([prediction[...,1:3],tf.exp(prediction[...,3:5])*tf.cast(anchors,dtype=tf.float32)],axis=-1)
    iouScore = iou(box_pred[isObj], target[..., 1:5][isObj])
    objloss = self.mse(iouScore*target[...,0:1][isObj],prediction[...,0:1][isObj])
    totalloss = (self.lambda_noobj*noobjloss) + (self.lambda_class*classloss)  + (self.lambda_box*boxloss) + (self.lambda_obj*objloss)
    return totalloss
  
if __name__ == '__main__':
  ANCHORS = np.array(ANCHORS)
  S = np.array(S).reshape(3,1,1)
  scaled = S*ANCHORS
  loss = YOLOLoss()
  for i in range(3): 
    target = tf.Variable(np.zeros((1,S[0][0][0],S[0][0][0],3,6)),dtype=tf.float32)
    prediction = tf.Variable(np.zeros((1,S[0][0][0],S[0][0][0],3*(20+5))),dtype=tf.float32)
    score = loss(target,prediction,scaled[i])
    assert score.numpy() - 6.9314 < 1e-4
  print('all test passed')