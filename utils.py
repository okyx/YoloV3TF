import tensorflow as tf

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

  union = box1area+box2area - intersection
  
  return intersection/union

def nms():
  pass

def nms():
  pass

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
  
  print('all test passed')
  