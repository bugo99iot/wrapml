from wrapml.imports.vanilla import logger

# ML

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer, \
    roc_auc_score, matthews_corrcoef, cohen_kappa_score, zero_one_loss
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# NN

import tensorflow as tf

logger.debug("TensorFlow ready. Version:", tf.__version__)

from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.metrics import Precision, Recall, Accuracy, SparseCategoricalAccuracy, \
    TrueNegatives, TruePositives, FalseNegatives, FalsePositives

from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
