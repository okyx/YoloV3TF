import tensorflow as tf
import config

class ConvClass(tf.keras.Model):
  def __init__(self,bn_act=True,**kwargs):
    super().__init__()
    self.bn_act = bn_act
    self.conv = tf.keras.layers.Conv2D(**kwargs)
    self.bn = tf.keras.layers.BatchNormalization()
    self.leaky = tf.keras.layers.LeakyReLU(0.1)

  def __call__(self,input):
    if self.bn_act:
      return self.leaky(self.bn(self.conv(input)))
    else:
      return self.conv(input)
    
class ScaleClass(tf.keras.Model):
  def __init__(self,num_classes=20):
    super().__init__()
    self.num_class=num_classes

  def __call__(self,input):
    filter_size = 3*(self.num_class+5)
    return ConvClass(filters=filter_size,kernel_size=1,padding='same')(input)

class ResidualClass(tf.keras.Model):
  def __init__(self,use_residual = True, num_repeats=1):
    super().__init__()
    self.use_residual = True
    self.num_repeats = num_repeats
  
  def __call__(self,input):
    input_channel = input.shape[-1]
    x = input
    for _ in range(self.num_repeats):
      x = ConvClass(filters=input_channel//2,kernel_size=1,padding='same')(x)
      x = ConvClass(filters=input_channel,kernel_size=3,padding='same')(x)
      x = tf.keras.layers.Add()([input,x])
      input = x
    return x

  
class TFYoloModel(tf.keras.Model):
  def __init__(self,image_shape=(416,416,3),num_classes=20,config = None):
    super().__init__()
    self.image_input = tf.keras.Input(shape=image_shape)
    self.num_classes = num_classes
    self.config = config

  def call(self,input):
    if not self.config or not isinstance(self.config,list):
      return self.image_input
    x = input
    outputs = []
    route_skip = []

    for config in self.config:
      if isinstance(config,tuple):
        x = ConvClass(filters=config[0],kernel_size=config[1],padding='same',strides=config[2])(x)
      elif isinstance(config,int):
        x = ResidualClass(num_repeats=config)(x)
        if config == 8 :
          route_skip.append(x)
      elif isinstance(config,str):
        if config == 'S':
          output = ScaleClass(num_classes=self.num_classes)(x)
          outputs.append(output)
        elif config == 'U':
          x = tf.keras.layers.UpSampling2D(size=2)(x)
          skip = route_skip[-1]
          x = tf.keras.layers.Concatenate()([skip,x])
          route_skip.pop()
    return outputs

  def summary(self):
    model = tf.keras.models.Model(inputs=self.image_input, outputs=self.call(self.image_input))
    return model.summary()
    
  def build_graph(self):
    model = tf.keras.models.Model(inputs=self.image_input, outputs=self.call(self.image_input))
    return model
  
if __name__ == '__main__':
  model = TFYoloModel(config=config.config,num_classes=config.num_classes)
  print(model.summary())