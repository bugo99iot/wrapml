import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, Dense, BatchNormalization, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Lambda, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import one_hot
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
logger.info("TensorFlow ready. Version:", tf.__version__)
