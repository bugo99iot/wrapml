from wrapml.imports.learn import RandomForestClassifier, KNeighborsClassifier, AdaBoostClassifier, SVC, MLPClassifier, \
    XGBClassifier

LSTMCLASSIFIER_MODEL_NAME = 'LSTMClassifier'
CONV2DCLASSIFIER_MODEL_NAME = 'Conv2dClassifier'

DEFAULT_GRID_SEARCH_PARAMETERS = {type(RandomForestClassifier()).__name__: {'n_estimators': [50, 100, 200],
                                                                            'class_weight': ['balanced', None],
                                                                            'min_samples_split': [0.2, 0.8, 2],
                                                                            'min_samples_leaf': [0.05, 0.5, 1],
                                                                            'max_features': ['auto', 'sqrt', None]},
                                  type(KNeighborsClassifier()).__name__: {'n_neighbors': [2, 3, 5]},
                                  type(AdaBoostClassifier()).__name__: {},
                                  type(SVC()).__name__: {'kernel': ['rbf', 'linear', 'poly']},
                                  type(MLPClassifier()).__name__: {'hidden_layer_sizes': [(100, 20,), (100,), (50,)],
                                                                   'max_iter': [200, 400],
                                                                   'alpha': [0.0001, 0.0001, 0.001]},
                                  type(XGBClassifier()).__name__: {'booster': ['gbtree', 'dart'],
                                                                   'max_depth': [2, 6, 10]},

                                  LSTMCLASSIFIER_MODEL_NAME: {},
                                  CONV2DCLASSIFIER_MODEL_NAME: {}
                                  }
DEFAULT_TEST_SIZE = 0.3
DEFAULT_RANDOM_STATE = 0

CLASSIFICATION_TYPE_BINARY = 'binary'
CLASSIFICATION_TYPE_MULTICLASS = 'multiclass'
CLASSIFICATION_TYPE_MULTILABEL = 'multilabel'  # todo: not supported
CLASSIFICATION_TYPE_ALLOWED = [CLASSIFICATION_TYPE_BINARY, CLASSIFICATION_TYPE_MULTICLASS]
