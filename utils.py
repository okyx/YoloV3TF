import tensorflow as tf
import config
from Boundbox import BoundBox
import numpy as np
from pprint import pprint

def iou_zeropoint(box1,box2):
  '''
    box1 : merupakan box target (1,2)
    box2 : merupakan anchors (n,2)
    untuk mencari tahu anchor mana yang paling cocok
  '''
  min_width = tf.math.minimum(box1[...,0],box2[...,0])
  min_height = tf.math.minimum(box1[...,1],box2[...,1])
  intersection = min_width * min_height
  union = (box1[...,0] * box1[...,1]) +  (box2[...,0] * box2[...,1]) - intersection
  return intersection / union

def iou(box_target,box_pred,kind='corner'):
  '''
    box_target : merupakan box target (batch,4)
    box_pred : merupakan box prediksi (batch,4)
    kalo jenis korner maka susunan (x1,y1,x2,y2)
    klo midpoint harus diubah ke corner karna (x,y,w,h)
  '''
  if kind == 'midpoint':
    boxtarget_x1 = tf.cast(box_target[:,0:1],dtype=tf.float64) - (box_target[:,2:3]/2)
    boxtarget_y1 = tf.cast(box_target[:,1:2],dtype=tf.float64) - (box_target[:,3:4]/2)
    boxtarget_x2 = tf.cast(box_target[:,0:1],dtype=tf.float64) + (box_target[:,2:3]/2)
    boxtarget_y2 = tf.cast(box_target[:,1:2],dtype=tf.float64) + (box_target[:,3:4]/2)
    boxpred_x1 = tf.cast(box_pred[:,0:1],dtype=tf.float64) - (box_pred[:,2:3]/2)
    boxpred_y1 = tf.cast(box_pred[:,1:2],dtype=tf.float64) - (box_pred[:,3:4]/2)
    boxpred_x2 = tf.cast(box_pred[:,0:1],dtype=tf.float64) + (box_pred[:,2:3]/2)
    boxpred_y2 = tf.cast(box_pred[:,1:2],dtype=tf.float64) + (box_pred[:,3:4]/2)
  if kind == 'corner':
    boxtarget_x1 = box_target[:,0:1]
    boxtarget_y1 = box_target[:,1:2]
    boxtarget_x2 = box_target[:,2:3]
    boxtarget_y2 = box_target[:,3:4]
    boxpred_x1 = box_pred[:,0:1]
    boxpred_y1 = box_pred[:,1:2]
    boxpred_x2 = box_pred[:,2:3]
    boxpred_y2 = box_pred[:,3:4]
  x1 = tf.math.maximum(boxtarget_x1,boxpred_x1)
  y1 = tf.math.maximum(boxtarget_y1,boxpred_y1)
  x2 = tf.math.minimum(boxtarget_x2,boxpred_x2)
  y2 = tf.math.minimum(boxtarget_y2,boxpred_y2)

  intersection = tf.math.maximum((x2-x1),0) * tf.math.maximum((y2-y1),0)
  box1area = tf.math.abs((boxtarget_x2-boxtarget_x1) * (boxtarget_y2-boxtarget_y1))
  box2area = tf.math.abs((boxpred_x2-boxpred_x1) * (boxpred_y2-boxpred_y1))

  union = box1area + box2area - intersection
  
  return intersection/union

def nms(bboxes,nms_threshold):
  new_bboxes = []
  for _,boxes in enumerate(bboxes):
    # print(new_bboxes)
    while len(boxes):
      sort_indices = np.argsort([box.score for box in boxes])[::-1]
      highestBox = boxes.pop(sort_indices[0])
      try:
        new_bboxes[_].append(highestBox)
      except:
        new_bboxes.append([highestBox])
      np.delete(sort_indices,0)
      highestBoxPoint = tf.constant([[highestBox.xmin,highestBox.ymin,highestBox.xmax,highestBox.ymax]])
      
      indices = []
      for i,box in enumerate(boxes):
        boxPoint = tf.constant([[box.xmin,box.ymin,box.xmax,box.ymax]])
        if box.classes == highestBox.classes and iou(highestBoxPoint,boxPoint)>nms_threshold:
          indices.append(i)
      for indeks in indices[::-1]:
        boxes.pop(indeks)
  return new_bboxes
      

def map():
  pass

def getRealBox(outputs,anchors,THRESHOLD): #[(batch,13,13,75),(batch,26,26,3,75),(batch,52,52,75)]
  bboxes = []
  for scale,output in enumerate(outputs):
    shape = output.shape # 32 13,13,75
    output = tf.reshape(output,shape=[shape[0],shape[1],shape[2],3,shape[3]//3])# 32 13,13,3,25
    output = output.numpy()
    output[...,0:3] = tf.nn.sigmoid(output[...,0:3])
    anchor = tf.reshape(anchors[scale],shape=(1,1,1,3,2))
    output[...,3:5] = tf.exp(tf.cast(output[...,3:5],dtype=tf.float32))*tf.cast(anchor,dtype=tf.float32)
    grids = shape[1] #32,13,13,3,25
    for _ in range(shape[0]):
      for grid in range(grids**2):
        x = grid // grids
        y = grid % grids
        for nb_anchor in range(3):
          isobj = output[_,x,y,nb_anchor,:]
          if isobj[0]< THRESHOLD:
            continue
          x_new = (x+isobj[1])/grids * config.IMAGE_SIZE
          y_new = (y+isobj[2])/grids * config.IMAGE_SIZE
          w_new = isobj[3]* config.IMAGE_SIZE
          h_new = isobj[4]* config.IMAGE_SIZE
          kelas = tf.argmax(tf.nn.softmax(isobj[5:])).numpy()
          score = tf.math.reduce_max(tf.nn.softmax(isobj[5:])).numpy()
          box = BoundBox(x_new-w_new/2,y_new-h_new/2,x_new+w_new/2,y_new+h_new/2,kelas,score)
          try:
            bboxes[_].append(box)
          except: 
            bboxes.append([box])
  return bboxes

if __name__ == '__main__':
  iouzero = iou_zeropoint(tf.constant([[2,3]]),tf.constant([[3,2]]))
  assert iouzero.numpy() == 0.5
  
  box_target = tf.Variable([[0,0,4,4],[0,0,4,4],[0,0,4,4]])
  box_pred = tf.Variable([[2,2,6,6],[2,2,6,6],[2,2,6,6]])
  iouCorner = iou(box_target,box_pred,kind='corner')
  assert iouCorner.shape == (3,1)
  for i in iouCorner:
    assert i == 4/28
    
  box_target = tf.Variable([[2,2,4,4],[2,2,4,4],[2,2,4,4]])
  box_pred = tf.Variable([[4,4,4,4],[4,4,4,4],[4,4,4,4]])
  iouMid = iou(box_target,box_pred,kind='midpoint')
  assert iouMid.shape == (3,1)
  for i in iouMid:
    assert i == 4/28
  
  ANCHORS = np.array(config.ANCHORS)
  S = np.array(config.S).reshape(3,1,1)
  scaled = S*ANCHORS
  
  anc87 = np.array([5,-0.815,-0.5,-2.790874551737623,-2.490841385307815,0,0,0,0,0,0,100,0,0,0,0,0,0,0,0,0,0,0,0,0])
  anc871 = np.array([5,-0.815,-0.5,-1.928651041133744 , -2.005333569526114,0,0,0,0,0,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  anc1614 = np.array([5,0.465,1.13,-1.6232693915825622,-2.1945755691646425,0,0,0,0,0,0,100,0,0,0,0,0,0,0,0,0,0,0,0,0])
  anc16141 = np.array([5,0.465,1.13,-1.6922622630695137 , -1.225175011976539,0,0,0,0,0,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  anc3329 = np.array([5,-1.216,0.05,-1.7568007842070847,-1.3121863889661687,0,0,0,0,0,0,100,0,0,0,0,0,0,0,0,0,0,0,0,0])
  anc33291 = np.array([5,-1.216,0.05,-1.0636536036471396 , -1.4663370687934272,0,0,0,0,0,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  anc1 = np.zeros((2,13,13,3,25))
  anc2 = np.zeros((2,26,26,3,25))
  anc3 = np.zeros((2,52,52,3,25))
  anc1[0,8,7,2,:] = anc87
  anc1[0,8,7,1,:] = anc871
  anc2[0,16,14,2,:] = anc1614
  anc2[0,16,14,1,:] = anc16141
  anc3[0,33,29,2,:] = anc3329
  anc3[0,33,29,1,:] = anc33291
  anc1[1,8,7,2,:] = anc87
  anc2[1,16,14,2,:] = anc1614
  anc3[1,33,29,2,:] = anc3329
  anc1 = anc1.reshape((2,13,13,75))
  anc2 = anc2.reshape((2,26,26,75))
  anc3 = anc3.reshape((2,52,52,75))
  output = [tf.Variable(anc1),tf.Variable(anc2),tf.Variable(anc3)]
  bboxes = getRealBox(output,scaled,config.CLASS_THRESHOLD)
  new_bboxese = nms(bboxes,config.NMS_THRESHOLD)
  target = {'xmax': 415.168,
            'xmin': 116.48,
            'ymax': 410.8,
            'ymin': 61.36}
  assert len(new_bboxese[0])==2 and len(new_bboxese[1])==1
  for boxes in new_bboxese:
    for box in boxes:
      variable = vars(box)
      for key in target.keys():
        assert abs(target[key]-variable[key])<1
  print('all test passed')
  