class BoundBox:
  def __init__(self, xmin, ymin, xmax, ymax, classes,score):
    self.xmin = xmin
    self.ymin = ymin
    self.xmax = xmax
    self.ymax = ymax
    self.classes = classes
    self.score = score