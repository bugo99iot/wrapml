# Utils
from wrapml.imports.vanilla import Dict, Optional, List, pprint

# DS
from wrapml.imports.science import np, pd
from wrapml.plots import make_training_history_plot, make_confusion_plot, make_roc_plot

# ML
from wrapml.imports.learn import RandomForestClassifier, KNeighborsClassifier, AdaBoostClassifier, SVC, MLPClassifier, \
    XGBClassifier
from wrapml.imports.learn import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef, \
    cohen_kappa_score, roc_auc_score, zero_one_loss, make_scorer
from wrapml.imports.learn import confusion_matrix as make_confusion_matrix

from wrapml.imports.learn import pickle, StratifiedShuffleSplit
from wrapml.imports.learn import train_test_split
from wrapml.imports.learn import GridSearchCV

from wrapml.imports.learn import History
from wrapml.imports.learn import EarlyStopping

# Pre-processing
from wrapml.imports.learn import MinMaxScaler, OneHotEncoder, to_categorical
from wrapml.imports.learn import one_hot

# NN
from wrapml.imports.learn import Sequential, Dense, Dropout, LSTM, Conv2D, MaxPooling2D, Flatten
from wrapml.imports.learn import Precision, Recall, Accuracy, SparseCategoricalAccuracy
from wrapml.imports.learn import one_hot
from wrapml.imports.learn import K

# Internal
from wrapml.utils.logging import logger
from wrapml.exceptions import ModelNotTrainableException

# Parameters
from wrapml.constants import DEFAULT_GRID_SEARCH_PARAMETERS, CLASSIFICATION_TYPE_BINARY, CLASSIFICATION_TYPE_MULTICLASS
from wrapml.constants import DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE
from wrapml.constants import CONV2DCLASSIFIER_MODEL_NAME, LSTMCLASSIFIER_MODEL_NAME


# todo HIGH PRIO:
#  - add refit best model when search all
#  - add get confusion matrix from refitted model in sklearn
#  - preprocess y and x when making prediction
#  - option to use / not use OHE


# todo MED PRIO:
#  - add print plots (ROC curve, confusion matrix, overfit, underfit, plot to compare models)
#  - fix labels encoding
#  - class weight support
#  - add CNN tensorflow and beyond + rescale image 255 / https://www.tensorflow.org/tutorials/images/classification


# todo LOW PRIO
#  - add OneVsRestClassifier
#  - add multilabel support
#  - add cohen k-score for inter-annotators agreement
#  - dd automatic overfit / underfit detection
#  - feature selection
#  - add score given min probability for classifier which provide probability



class ClassificationTask:

    def __init__(self, x: np.ndarray,
                 y: np.ndarray):

        # todo: make some properties available in init?

        # setup
        self.random_state = DEFAULT_RANDOM_STATE
        self.n_jobs = -1

        # scoring
        self.test_size = DEFAULT_TEST_SIZE
        self.stratify = True  # todo: add stratify = False option
        self.k_fold_cross_validation = 1
        self.fit_time_best_model_k_folds_s: Optional[int] = None
        self.fit_time_total_s: Optional[int] = None
        self.score: Dict = {}
        self.score_dp: int = 4
        self.score_average_method: str = 'weighted'
        self.do_grid_search: bool = False
        self.return_train_score: bool = True
        self.scoring_metrics_keras: Dict = {}
        self.scoring_metrics_tensorflow: Dict = {}
        self.score_criteria_for_best_model_fit = 'f1_score'

        # model
        self.model = None
        self.best_model_parameters: Dict = {}
        self.model_name: Optional[str] = None

        self.best_parameters_from_grid_search: Dict = {}
        self.parameter_combinations_searched: Optional[int] = None

        self.history: Optional[History] = None

        # report
        self.report: Dict = {}
        self.small_report: Dict = {}

        # data
        self.y_input = y
        self.y: np.ndarray = self.y_input
        self.x_input = x
        self.x: np.ndarray = self.x_input

        if not isinstance(self.y, np.ndarray):
            raise Exception('y must be a numpy array')

        if not isinstance(self.x, np.ndarray):
            raise Exception('x must be a numpy array')

        if pd.isna(self.y).any():
            raise Exception('y cannot contain nan values')

        if pd.isna(self.y).any():
            raise Exception('x cannot contain nan values')

        # todo:
        #  - now we OHE y always: add support with y labels are ordinal and OHE should be avoided
        #  - if not doing OHE, need to convert categorical labels
        # check y
        self.y_shape = self.y.shape
        self.y_dim = len(self.y_shape)

        if self.y_dim not in (1, 2) or self.y_shape[0] < 10 or (self.y_dim == 2 and self.y_shape[1] != 1):
            raise Exception('y should be a numpy array with shape (n,) or (n, 1), n >= 10 (multi-label is not supported)')

        if self.y_dim == 1:
            self.y_dim1 = self.y
            self.y_dim1_shape = self.y_dim1.shape
            self.y_dim1_dim = len(self.y_dim1_shape)
            self.y_dim2: np.ndarray = self.y.reshape(self.y_shape[0], 1)
            self.y_dim2_shape = self.y_dim2.shape
            self.y_dim2_dim = len(self.y_dim2_shape)
        elif self.y_dim == 2:
            self.y_dim1 = self.y.reshape((self.y_shape[0], ))
            self.y_dim1_shape = self.y_dim1.shape
            self.y_dim1_dim = len(self.y_dim1_shape)
            self.y_dim2: np.ndarray = self.y
            self.y_dim2_shape = self.y_dim2.shape
            self.y_dim2_dim = len(self.y_dim2_shape)

        # at this point, self.y has shape (n, 1), n >= 10

        # self.labels: List[str] = list(set([k for k in self.y_dim1]))
        # self.labels.sort()

        self.ohe = OneHotEncoder(categories='auto', sparse=False)
        self.y_ohe = self.ohe.fit_transform(self.y_dim2)
        self.labels = list(self.ohe.categories_[0])

        self.y_ohe_shape = self.y_ohe.shape
        self.y_ohe_dim = len(self.y_ohe.shape)

        self.n_classes: int = len(self.labels)  # or = self.y_ohe.shape[1]

        if not (all(isinstance(k, int) for k in self.labels) or all(isinstance(k, str) for k in self.labels)
                or all(isinstance(k, np.integer) for k in self.labels)):
            raise Exception('input labels must be int or str')

        # todo: add multi-label support
        if self.n_classes > 2:
            self.classification_type = CLASSIFICATION_TYPE_MULTICLASS
        elif self.n_classes == 2:
            self.classification_type = CLASSIFICATION_TYPE_BINARY
        else:
            raise Exception('n_classes must be 2 or larger')

        # check x
        self.x_shape = self.x.shape
        self.x_dim = len(self.x.shape)

        if self.x_dim not in (1, 2, 3, 4) or self.x_shape[0] < 10:
            raise Exception('x should be a numpy array with shape (n, m) or (n, m, p) or (n, m, p, g), n >= 10')

        # todo: add possibility to convert non-images into images to run 2D CNN on it

        if len(self.x_shape) == 1:
            # (n_samples, 1), meaning n data points, each one with a scalar measure,
            # e.g. measuring temperature at timestamps

            self.x_dim2 = self.x.reshape(self.x_shape[0], 1)
            self.x_dim2_shape = self.x_dim2.shape
            self.x_dim2_dim = len(self.x_dim2.shape)

            self.x_dim3 = self.x_dim2.reshape((self.x_dim2_shape[0], 1, 1))
            self.x_dim3_shape = self.x_dim3.shape
            self.x_dim3_dim = len(self.x_dim3.shape)

            self.x_dim4 = self.x_dim2.reshape((self.x_dim2_shape[0], 1, 1, 1))
            self.x_dim4_shape = self.x_dim4.shape
            self.x_dim4_dim = len(self.x_dim4.shape)

        if len(self.x_shape) == 2:
            # (n_samples, m), meaning n data points, each one with a vector measure
            # e.g. measuring temperature, wind, precipitation at timestamps

            self.x_dim2 = self.x
            self.x_dim2_shape = self.x_dim2.shape
            self.x_dim2_dim = len(self.x_dim2.shape)

            self.x_dim3 = self.x_dim2.reshape((self.x_dim2_shape[0], self.x_dim2_shape[1], 1))
            self.x_dim3_shape = self.x_dim3.shape
            self.x_dim3_dim = len(self.x_dim3.shape)

            self.x_dim4 = self.x_dim2.reshape((self.x_dim2_shape[0], self.x_dim2_shape[1], 1, 1))
            self.x_dim4_shape = self.x_dim4.shape
            self.x_dim4_dim = len(self.x_dim4.shape)

        elif len(self.x_shape) == 3:
            # (n_samples, m, n),
            # meaning: either n data points, each one with multiple vector measures
            # e.g. accelerometer in 3d at timestamps
            # or: image with no channels, e.g. grayscale with one dimension missing

            self.x_dim3 = self.x
            self.x_dim3_shape = self.x_dim3.shape
            self.x_dim3_dim = len(self.x_dim3.shape)

            self.x_dim2 = self.x_dim3.reshape((self.x_dim3_shape[0], self.x_dim3_shape[1] * self.x_dim3_shape[2]))
            self.x_dim2_shape = self.x_dim2.shape
            self.x_dim2_dim = len(self.x_dim2.shape)

            # channel last
            self.x_dim4 = self.x_dim3.reshape((self.x_dim3_shape[0], self.x_dim3_shape[1], self.x_dim3_shape[2], 1))
            self.x_dim4_shape = self.x_dim4.shape
            self.x_dim4_dim = len(self.x_dim4.shape)

        elif len(self.x_shape) == 4:
            # (n_samples, xdim, ydim, n_channels)
            # image with channels last

            logger.warning('If image data passes, we assume channels last ordering, '
                           'e.g. (n_samples, xdim, ydim, n_channels)')

            self.x_dim4 = self.x
            self.x_dim4_shape = self.x_dim4.shape
            self.x_dim4_dim = len(self.x_dim4.shape)

            self.x_dim2 = self.x_dim4.reshape((self.x_dim4_shape[0], self.x_dim4_shape[1] * self.x_dim4_shape[2] * self.x_dim4_shape[3]))
            self.x_dim2_shape = self.x_dim2.shape
            self.x_dim2_dim = len(self.x_dim2.shape)

            self.x_dim3 = self.x_dim4.reshape((self.x_dim4_shape[0], self.x_dim4_shape[1] * self.x_dim4_shape[2], self.x_dim4_shape[3]))
            self.x_dim3_shape = self.x_dim3.shape
            self.x_dim3_dim = len(self.x_dim3.shape)

        else:
            raise Exception('This should never happen')

        self.x_dim2 = self.x_dim2.astype('float32')
        self.x_dim3 = self.x_dim3.astype('float32')
        self.x_dim4 = self.x_dim4.astype('float32')

        # todo: add rescale option

        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        self.y_test_pred, self.y_test_pred_proba = None, None

    # Keras based

    def train_with_knn(self,
                       do_grid_search: bool = None,
                       grid_search_parameters: Dict = None,
                       score_criteria_for_best_model_fit: str = None,
                       **kwargs
                       ):

        model = KNeighborsClassifier(**kwargs, n_jobs=self.n_jobs)

        self._train_with_keras(model=model,
                               grid_search_parameters=grid_search_parameters,
                               score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                               do_grid_search=do_grid_search)

    def train_with_random_forests(self,
                                  do_grid_search: bool = None,
                                  grid_search_parameters: Dict = None,
                                  score_criteria_for_best_model_fit: str = None,
                                  **kwargs):

        model = RandomForestClassifier(**kwargs, random_state=self.random_state, n_jobs=self.n_jobs)

        self._train_with_keras(model=model,
                               grid_search_parameters=grid_search_parameters,
                               score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                               do_grid_search=do_grid_search)

    def train_with_mlp(self,
                       do_grid_search: bool = None,
                       grid_search_parameters: Dict = None,
                       score_criteria_for_best_model_fit: str = None,
                       **kwargs):

        model = MLPClassifier(**kwargs, random_state=self.random_state, early_stopping=True)

        self._train_with_keras(model=model,
                               grid_search_parameters=grid_search_parameters,
                               score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                               do_grid_search=do_grid_search)

    def train_with_xgboost(self,
                           do_grid_search: bool = None,
                           grid_search_parameters: Dict = None,
                           score_criteria_for_best_model_fit: str = None,
                           **kwargs):
        # https://xgboost.readthedocs.io/en/latest/python/python_api.html
        # https://xgboost.readthedocs.io/en/latest/parameter.html

        model = XGBClassifier(**kwargs, n_jobs=self.n_jobs, random_state=self.random_state)

        self._train_with_keras(model=model,
                               grid_search_parameters=grid_search_parameters,
                               score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                               do_grid_search=do_grid_search)

    def train_with_ada(self,
                       do_grid_search: bool = None,
                       grid_search_parameters: Dict = None,
                       score_criteria_for_best_model_fit: str = None,
                       **kwargs):

        model = AdaBoostClassifier(**kwargs, random_state=self.random_state)

        self._train_with_keras(model=model,
                               grid_search_parameters=grid_search_parameters,
                               score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                               do_grid_search=do_grid_search)

    def train_with_svc(self,
                       do_grid_search: bool = None,
                       grid_search_parameters: Dict = None,
                       score_criteria_for_best_model_fit: str = None,
                       **kwargs):

        model = SVC(**kwargs, random_state=self.random_state)

        self._train_with_keras(model=model,
                               grid_search_parameters=grid_search_parameters,
                               score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                               do_grid_search=do_grid_search)

    def _train_with_keras(self,
                          model,
                          do_grid_search: bool,
                          grid_search_parameters: Dict,
                          score_criteria_for_best_model_fit: str
                          ):
        """

        :param model:
        :param grid_search_parameters:

        :return:
        """

        self._init_training()

        model = model
        self.model_name = type(model).__name__

        logger.info('Training with {}.'.format(self.model_name))

        if do_grid_search is not None:
            do_grid_search_for_model = do_grid_search
        else:
            do_grid_search_for_model = self.do_grid_search

        if not do_grid_search_for_model:
            grid_search_parameters = {}
        else:
            if grid_search_parameters:
                pass
            else:
                grid_search_parameters = DEFAULT_GRID_SEARCH_PARAMETERS[self.model_name]

        score_criteria_for_best_model_fit = score_criteria_for_best_model_fit \
            if score_criteria_for_best_model_fit else self.score_criteria_for_best_model_fit

        logger.debug('Using StratifiedShuffleSplit for cross-validation.')
        sss = StratifiedShuffleSplit(n_splits=self.k_fold_cross_validation,
                                     test_size=self.test_size,
                                     random_state=self.random_state)

        model_gridsearch = GridSearchCV(estimator=model,
                                        param_grid=grid_search_parameters,
                                        n_jobs=self.n_jobs,
                                        scoring=self.scoring_metrics_keras,
                                        refit=score_criteria_for_best_model_fit,
                                        cv=sss,
                                        return_train_score=True)

        model_gridsearch.fit(self.x_dim2, self.y_dim1)

        # todo: maybe make refit best model manually so that confusion matrix can be trusted

        for train_index, test_index in sss.split(self.x_dim2, self.y_dim1):

            self.x_train, self.x_test = self.x_dim2[train_index], self.x_dim2[test_index]
            self.y_train, self.y_test = self.y_dim1[train_index], self.y_dim1[test_index]
            continue

        # get one test split for confusion matrix
        self.y_test_pred = model_gridsearch.best_estimator_.predict(self.x_test)
        try:
            self.y_test_pred_proba = model_gridsearch.best_estimator_.predict_proba(self.x_test)
        except:
            self.y_test_pred_proba = None

        self.confusion_matrix: np.ndarray = make_confusion_matrix(y_true=self.y_test, y_pred=self.y_test_pred,
                                                                  labels=self.labels)

        index_of_best_model = model_gridsearch.best_index_
        results = model_gridsearch.cv_results_

        # let's parse score for best model, mean means mean of k-folds
        for j in ['train', 'test']:
            set_score = {}
            for k in self.scoring_metrics_keras.keys():
                set_score[k + '_mean'] = round(float(results['mean_' + j + '_' + k][index_of_best_model]), self.score_dp)
                set_score[k + '_std'] = round(float(results['std_' + j + '_' + k][index_of_best_model]), self.score_dp)
            self.score[j] = set_score

        self.score['best_score_criteria'] = score_criteria_for_best_model_fit
        self.score['best_score'] = self.score['test'][score_criteria_for_best_model_fit + '_mean']
        if round(float(model_gridsearch.best_score_), self.score_dp) != self.score['best_score']:
            raise Exception('could not identify best score')

        self.best_parameters_from_grid_search = model_gridsearch.best_params_
        self.parameter_combinations_searched = len(results['mean_fit_time'])
        self.model = model_gridsearch.best_estimator_
        self.model_name = type(model).__name__
        self.best_model_parameters = self.model.get_params()

        self.fit_time_best_model_k_folds_s = round(float(model_gridsearch.refit_time_), 4)
        self.fit_time_total_s = round(float(np.sum(results['mean_fit_time'])), 4)

        logger.info('Training with {} done. Fit time: {}s.'.format(self.model_name,
                                                                   self.fit_time_total_s))

        self._calculate_report_for_model_keras()

    def make_roc_plot(self, pos_label: int or str):
        if self.classification_type != CLASSIFICATION_TYPE_BINARY:
            raise Exception('roc curve supported for binary classification problems only')
        if self.y_test_pred_proba is None:
            raise Exception('cannot plot roc curve for model {}, probability not available'.format(self.model_name))

        make_roc_plot(y_test=self.y_test, y_test_pred_prob=self.y_test_pred_proba,
                      pos_label=pos_label)

    def make_confusion_plot(self, normalize: bool = False):
        if self.confusion_matrix is None:
            raise Exception('Cannot make confusion plot, confusion matrix has not been instantiated')
        make_confusion_plot(confusion_matrix=self.confusion_matrix,
                            labels=self.labels,
                            normalize=normalize,
                            model_name=self.model_name)

    def _calculate_report_for_model_keras(self):

        report = self.score

        for i in ['fit_time_best_model_k_folds_s',
                  'fit_time_total_s',
                  'best_parameters_from_grid_search',
                  'parameter_combinations_searched',
                  'k_fold_cross_validation',
                  'n_classes',
                  'classification_type',
                  'model_name',
                  'best_model_parameters'
                  ]:
            report[i] = self.__getattribute__(i)

        self.report = report

    def print_report(self):
        if not self.report:
            raise Exception('Cannot print report, report has not been instantiated')
        pprint(self.report)

    def print_small_report(self):
        if not self.report:
            raise Exception('Cannot print small report, small report has not been instantiated')
        pprint(self.small_report)

    # Tensorflow based

    def train_with_lstm(self,
                        **kwargs):

        # todo:
        #  - add k-folds
        #  - add k-folds + grid search: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
        #  - add report
        #  - loss as binary crossentropy + 1 neuron sigmoid final layer when binary

        model_name = LSTMCLASSIFIER_MODEL_NAME

        model = Sequential(name=self.model_name)
        model.add(LSTM(units=32, input_shape=(self.x_dim3_shape[1], self.x_dim3_shape[2])))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=100, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))

        # parse kwargs
        early_stopping_callback = EarlyStopping(monitor='loss', patience=10) if kwargs.get('early_stopping') else None
        epochs = kwargs.get('epochs') if kwargs.get('epochs') else 100
        if epochs < 10:
            raise Exception('epochs must be >= 10')
        batch_size = kwargs.get('batch_size') if kwargs.get('batch_size') else 32

        callbacks = [early_stopping_callback] if early_stopping_callback else []

        self._train_with_tensorflow(model=model,
                                    model_name=model_name,
                                    epochs=epochs,
                                    callbacks=callbacks,
                                    batch_size=batch_size,
                                    x=self.x_dim3,
                                    y=self.y_ohe)

    def train_with_conv2d(self,
                          **kwargs):

        model_name = CONV2DCLASSIFIER_MODEL_NAME

        if self.x_dim4_shape[1] != self.x_dim4_shape[2]:
            raise ModelNotTrainableException('Cannot train with {} given x shape'.format(self.model_name))

        # todo:
        #  - infer input channels first or last and squared
        #  - infer batch size and 2d kernels from image size
        """
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
        """

        model = Sequential(name=self.model_name)
        model.add(Conv2D(filters=32,
                         kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(self.x_dim4_shape[1], self.x_dim4_shape[2], self.x_dim4_shape[3])))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64,
                         kernel_size=(11, 11),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(units=100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='softmax'))

        # parse kwargs
        early_stopping_callback = EarlyStopping(monitor='loss', patience=10) if kwargs.get('early_stopping') else None
        epochs = kwargs.get('epochs') if kwargs.get('epochs') else 100
        if epochs < 10:
            raise Exception('epochs must be >= 10')
        batch_size = kwargs.get('batch_size') if kwargs.get('batch_size') else 32

        callbacks = [early_stopping_callback] if early_stopping_callback else []

        self._train_with_tensorflow(model=model,
                                    model_name=model_name,
                                    epochs=epochs,
                                    callbacks=callbacks,
                                    batch_size=batch_size,
                                    x=self.x_dim4,
                                    y=self.y_ohe)

    def _train_with_tensorflow(self,
                               model,
                               model_name: str,
                               x,
                               y,
                               epochs: Optional[int],
                               callbacks: Optional[List],
                               batch_size: Optional[int],
                               ):
        self._init_training()

        self.model_name = model_name
        self.model = model

        self.model.summary()

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=self.scoring_metrics_tensorflow
        )

        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state,
                                                            stratify=self.y_ohe,
                                                            )

        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=True,
            callbacks=callbacks
        )

        self.score = self.model.evaluate(x_test, y_test)

        # get back list of original labels, e.g. ['cat', 'dog', etc...]
        y_test_labels = [k[0] for k in self.ohe.inverse_transform(y_test)]
        y_test_pred = self.model.predict(x_test)
        y_test_pred_labels = [k[0] for k in self.ohe.inverse_transform(y_test_pred)]

        self.confusion_matrix: np.ndarray = make_confusion_matrix(y_true=y_test_labels,
                                                                  y_pred=y_test_pred_labels,
                                                                  labels=self.labels)

        # todo: get test score and calculate proper report
        self._calculate_report_for_model_tensorflow()

        return

    def make_training_history_plot(self):

        make_training_history_plot(history=self.history, metric='accuracy', model_name=self.model_name)
        make_training_history_plot(history=self.history, metric='loss', model_name=self.model_name)

    def _calculate_report_for_model_tensorflow(self):
        # todo: calculate report
        self.report = self.score

    def predict(self, x: np.ndarray):
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray):
        # some models might now return a probability
        if self.model_name == [type(SVC).__name__]:

            array_nan = np.empty((x.shape[0]))

            array_nan[:] = np.NaN
            return array_nan
        return self.model.predict_proba(x)

    def save_model(self, path: str):
        pickle.dump(self.model, open(path, 'wb'))

    def _init_training(self):
        self._init_scoring()
        self.fit_time_total_s = None
        self.fit_time_best_model_k_folds_s = None
        self.score = {}
        self.model = None
        self.model_name = None
        self.best_model_parameters = {}
        self.best_parameters_from_grid_search = {}
        self.parameter_combinations_searched = None
        self.report = {}
        self.small_report = {}
        self.confusion_matrix = None
        self.history = None
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        self.y_test_pred, self.y_test_pred_proba = None, None

    def _init_scoring(self):
        # todo: change scoring technique depending wther probelm is binary or multiclass
        logger.debug('returning 0 on precision score zerodivision')

        self.scoring_metrics_keras = {'accuracy_score': make_scorer(accuracy_score),
                                      'f1_score': make_scorer(f1_score, average=self.score_average_method),
                                      'precision_score': make_scorer(precision_score, average=self.score_average_method,
                                                                     zero_division=0),
                                      'recall_score': make_scorer(recall_score, average=self.score_average_method),
                                      'matthews_score': make_scorer(matthews_corrcoef),
                                      'zero_one_loss_score': make_scorer(zero_one_loss),
                                      # 'roc_auc_score': make_scorer(roc_auc_score, average='macro', multi_class='ovo')
                                      }
        self.scoring_metrics_tensorflow = ['accuracy', Precision(thresholds=0.5), Recall(thresholds=0.5)]

    def search_estimator(self,
                         do_grid_search: bool = False):

        # todo: add custom scoring criteria?

        report = {}

        self.train_with_random_forests(do_grid_search=do_grid_search)
        report[self.model_name] = self.report

        self.train_with_knn(do_grid_search=do_grid_search)
        report[self.model_name] = self.report

        self.train_with_mlp(do_grid_search=do_grid_search)
        report[self.model_name] = self.report

        self.train_with_svc(do_grid_search=do_grid_search)
        report[self.model_name] = self.report

        self.train_with_xgboost(do_grid_search=do_grid_search)
        report[self.model_name] = self.report

        self.train_with_ada(do_grid_search=do_grid_search)
        report[self.model_name] = self.report

        best_estimator_best_score = None
        best_estimator_name = None
        best_estimator_best_score_criteria = None
        best_estimator_fit_time_k_folds_s = None
        for k, v in report.items():

            if best_estimator_best_score is None:
                best_estimator_best_score = v['best_score']
                best_estimator_best_score_criteria = v['best_score_criteria']
                best_estimator_name = k
                best_estimator_fit_time_k_folds_s = v['fit_time_best_model_k_folds_s']

            elif v['best_score'] > best_estimator_best_score:
                best_estimator_best_score = v['best_score']
                best_estimator_best_score_criteria = v['best_score_criteria']
                best_estimator_name = k
                best_estimator_fit_time_k_folds_s = v['fit_time_best_model_k_folds_s']

        report['best_estimator'] = {}
        report['best_estimator']['best_estimator_name'] = best_estimator_name
        report['best_estimator']['best_estimator_best_score'] = best_estimator_best_score
        report['best_estimator']['best_estimator_best_score_criteria'] = best_estimator_best_score_criteria
        report['best_estimator']['best_estimator_fit_time_k_folds_s'] = best_estimator_fit_time_k_folds_s

        small_report = {'best_estimator': report['best_estimator']}

        self.report = report
        self.small_report = small_report
