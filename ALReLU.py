  
"""
Python (Keras) implemenation of ALReLU activation function used in 
"ALReLU: A different approach on Leaky ReLU activation function
to improve Neural Networks Performance" 
https://arxiv.org/abs/2012.07564?context=cs.LG
"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Lambda
from tensorflow.keras.utils import get_custom_objects
def ALReLU(x):
  """
  The gradients are automatically calculated on TF2
  """
  alpha = 0.01
  return K.maximum(K.abs(alpha*x), x)
  
#usage between convolution layers
get_custom_objects().update({'ALReLU':
tf.keras.layers.Activation(ALReLU)})
conv = Conv2D(32, (5, 5))(visible)
conv_act = ALReLU(conv)
conv_act_batch = BatchNormalization()(conv_act)
conv_maxpool = MaxPooling2D()(conv_act_batch)
conv_dropout = Dropout(0.1)(conv_maxpool)
