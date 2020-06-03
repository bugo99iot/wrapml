# Utils
from wrapml.imports.vanilla import Dict, Optional, List, pprint, datetime

# DS
from wrapml.imports.science import np, pd
from wrapml.plots import make_training_history_plot, make_confusion_plot, make_roc_plot

# ML
from wrapml.imports.learn import RandomForestClassifier, KNeighborsClassifier, AdaBoostClassifier, SVC, MLPClassifier, \
    XGBClassifier
from wrapml.imports.learn import MODEL_CLASSES_NOT_SUPPORTING_PARALLEL_JOBS, MODEL_CLASSES_NOT_SUPPORTING_RANDOM_STATE
from wrapml.imports.learn import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef, \
    cohen_kappa_score, roc_auc_score, zero_one_loss, auc, make_scorer
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
from wrapml.imports.learn import tf_precision_score, tf_recall_score, tf_f1_score_funk, tf_matthews_score_funk, tf_auc_score
from wrapml.imports.learn import KerasClassifier as TensorflowToKerasClassifier
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
#  - check if timeseries autocorrelates (check if it's timeseries)
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
        self.score: Dict = {}
        self.score_dp: int = 4
        self.score_average_method: str = 'weighted'
        self.return_train_score: bool = True
        self.scoring_metrics_keras_scorers: Dict = {}
        self.scoring_metrics_keras_scorers_for_gridsearch: Dict = {}
        self.scoring_metrics_tensorflow: Dict = {}
        self.scoring_metrics_tensorflow_ordered: List = []
        self.score_criteria_for_best_model_fit = 'recall_score'

        # model
        self.model = None
        self.best_model_parameters: Dict = {}
        self.model_name: Optional[str] = None

        self.history: Optional[History] = None

        # report
        self.report: Dict = {}

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

        if self.x_dim == 1:
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

        elif self.x_dim == 2:
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

        elif self.x_dim == 3:
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

        elif self.x_dim == 4:
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
            raise Exception('x_dim not allowed')

        self.x_dim2 = self.x_dim2.astype('float32')
        self.x_dim3 = self.x_dim3.astype('float32')
        self.x_dim4 = self.x_dim4.astype('float32')

        # todo: add rescale option

        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        self.y_test_pred, self.y_test_pred_proba = None, None

    # Keras based

    def train_with_knn(self,
                       do_grid_search: bool = False,
                       grid_search_parameters: Dict = None,
                       score_criteria_for_best_model_fit: str = None,
                       k_folds: int = 1,
                       **kwargs
                       ):

        model = KNeighborsClassifier

        self._train_with_keras(model_class=model,
                               grid_search_parameters=grid_search_parameters,
                               score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                               do_grid_search=do_grid_search,
                               model_parameters=kwargs,
                               k_folds=k_folds)

    def train_with_random_forests(self,
                                  do_grid_search: bool = False,
                                  grid_search_parameters: Dict = None,
                                  score_criteria_for_best_model_fit: str = None,
                                  k_folds: int = 1,
                                  **kwargs):

        model = RandomForestClassifier

        self._train_with_keras(model_class=model,
                               grid_search_parameters=grid_search_parameters,
                               score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                               do_grid_search=do_grid_search,
                               model_parameters=kwargs,
                               k_folds=k_folds)

    def train_with_mlp(self,
                       do_grid_search: bool = False,
                       grid_search_parameters: Dict = None,
                       score_criteria_for_best_model_fit: str = None,
                       k_folds: int = 1,
                       **kwargs):

        model = MLPClassifier

        self._train_with_keras(model_class=model,
                               grid_search_parameters=grid_search_parameters,
                               score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                               do_grid_search=do_grid_search,
                               model_parameters=kwargs,
                               k_folds=k_folds)

    def train_with_xgboost(self,
                           do_grid_search: bool = False,
                           grid_search_parameters: Dict = None,
                           score_criteria_for_best_model_fit: str = None,
                           k_folds: int = 1,
                           **kwargs):
        # https://xgboost.readthedocs.io/en/latest/python/python_api.html
        # https://xgboost.readthedocs.io/en/latest/parameter.html

        model = XGBClassifier

        self._train_with_keras(model_class=model,
                               grid_search_parameters=grid_search_parameters,
                               score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                               do_grid_search=do_grid_search,
                               model_parameters=kwargs,
                               k_folds=k_folds)

    def train_with_ada(self,
                       do_grid_search: bool = False,
                       grid_search_parameters: Dict = None,
                       score_criteria_for_best_model_fit: str = None,
                       k_folds: int = 1,
                       **kwargs):

        model = AdaBoostClassifier

        self._train_with_keras(model_class=model,
                               grid_search_parameters=grid_search_parameters,
                               score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                               do_grid_search=do_grid_search,
                               model_parameters=kwargs,
                               k_folds=k_folds)

    def train_with_svc(self,
                       do_grid_search: bool = False,
                       grid_search_parameters: Dict = None,
                       score_criteria_for_best_model_fit: str = None,
                       k_folds: int = 1,
                       **kwargs):

        model = SVC

        self._train_with_keras(model_class=model,
                               grid_search_parameters=grid_search_parameters,
                               score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                               do_grid_search=do_grid_search,
                               model_parameters=kwargs,
                               k_folds=k_folds)

    def _train_with_keras(self,
                          model_class,
                          do_grid_search: bool,
                          grid_search_parameters: Dict,
                          score_criteria_for_best_model_fit: str,
                          model_parameters: Dict,
                          k_folds: int,
                          ):
        """

        :return:
        """

        self._init_training()

        # parse arguments

        self.model_name = model_class.__name__

        logger.info('Training with {}.'.format(self.model_name))

        # check k_folds for cross validation

        if not isinstance(k_folds, int):
            raise Exception('k_folds must be int')

        # check do grid search

        if not isinstance(do_grid_search, bool):
            raise Exception('do_grid_search must be bool')

        # parse score criteria for best model fit

        score_criteria_for_best_model_fit = score_criteria_for_best_model_fit \
            if score_criteria_for_best_model_fit else self.score_criteria_for_best_model_fit

        # parse model parameters

        if do_grid_search:
            model_parameters = {}
        else:
            model_parameters = model_parameters if model_parameters is not None else {}

        if model_class in MODEL_CLASSES_NOT_SUPPORTING_PARALLEL_JOBS:
            if 'n_jobs' in model_parameters.keys():
                model_parameters.pop('n_jobs')
        else:
            if 'n_jobs' in model_parameters.keys():
                pass
            else:
                model_parameters['n_jobs'] = self.n_jobs

        if model_class in MODEL_CLASSES_NOT_SUPPORTING_RANDOM_STATE:
            if 'random_state' in model_parameters.keys():
                model_parameters.pop('random_state')
        else:
            if 'random_state' in model_parameters.keys():
                pass
            else:
                model_parameters['random_state'] = self.random_state

        # instantiating splitter for cross-validation

        logger.debug('Using StratifiedShuffleSplit for cross-validation.')
        sss = StratifiedShuffleSplit(n_splits=k_folds,
                                     test_size=self.test_size,
                                     random_state=self.random_state)

        self.score['k_folds'] = k_folds

        self.score['best_score_criteria'] = score_criteria_for_best_model_fit

        if do_grid_search:

            if grid_search_parameters:
                pass
            else:
                grid_search_parameters = DEFAULT_GRID_SEARCH_PARAMETERS[self.model_name]

            gridsearch = GridSearchCV(estimator=model_class(**model_parameters),
                                      param_grid=grid_search_parameters,
                                      n_jobs=self.n_jobs,
                                      scoring=self.scoring_metrics_keras_scorers_for_gridsearch,
                                      cv=sss,
                                      refit=False,
                                      return_train_score=True)

            gridsearch_start = datetime.datetime.now()
            gridsearch.fit(self.x_dim2, self.y_dim1)
            time_taken_in_s: int = (datetime.datetime.now() - gridsearch_start).seconds
            self.score['time_taken_total_fit_in_s'] = time_taken_in_s

            gridsearch_results: Dict = gridsearch.cv_results_

            best_estimator_index: int = \
                int(np.argmax(gridsearch_results['mean_test_' + score_criteria_for_best_model_fit]))
            best_parameters_from_grid_search: Dict = gridsearch_results['params'][best_estimator_index]
            if 'random_state' in model_parameters:
                best_parameters_from_grid_search['random_state'] = model_parameters['random_state']
            self.score['model_parameters'] = best_parameters_from_grid_search
            self.score['parameters_combinations_searched'] = len(gridsearch_results['params'])

            # let's parse score for best model, mean of k-folds
            for j in ['train', 'test']:
                set_score = {}
                for k in self.scoring_metrics_keras_scorers.keys():
                    set_score[k + '_k_fold_mean'] = \
                        round(float(gridsearch_results['mean_' + j + '_' + k][best_estimator_index]), self.score_dp)
                    set_score[k + '_k_fold_std'] = \
                        round(float(gridsearch_results['std_' + j + '_' + k][best_estimator_index]), self.score_dp)
                self.score[j] = set_score

            self.score['best_score_k_fold_mean'] = self.score['test'][score_criteria_for_best_model_fit + '_k_fold_mean']
            self.score['best_score_k_fold_std'] = self.score['test'][score_criteria_for_best_model_fit + '_k_fold_std']

            # refit best model with k fold
            self.model = model_class(**best_parameters_from_grid_search)

            for train_index, test_index in sss.split(self.x_dim2, self.y_dim1):
                self.x_train, self.x_test = self.x_dim2[train_index], self.x_dim2[test_index]
                self.y_train, self.y_test = self.y_dim1[train_index], self.y_dim1[test_index]

                continue

            start_refit = datetime.datetime.now()
            self.model.fit(self.x_train, self.y_train)
            time_taken_in_s: int = (datetime.datetime.now() - start_refit).seconds
            self.score['time_taken_1_fold_fit_in_s'] = time_taken_in_s

            # get predictions
            self.y_test_pred = self.model.predict(self.x_test)

            for k, v in self.scoring_metrics_keras_scorers.items():
                self.score['test'][k + '_1_fold'] = round(v['scorer'](y_true=self.y_test,
                                                                          y_pred=self.y_test_pred,
                                                                          **v['kwargs']), self.score_dp)

            self.score['best_score_1_fold'] = self.score['test'][score_criteria_for_best_model_fit + '_1_fold']

        else:

            self.model = model_class(**model_parameters)

            self.score['model_parameters'] = {k: v for k, v in model_parameters.items() if k != 'n_jobs'}
            self.score['parameters_combinations_searched'] = None

            all_fold_scores = []
            # we reverse so that the last spit is the first one and self.x_test, self.y_test are same as step above
            for train_index, test_index in reversed(list(sss.split(self.x_dim2, self.y_dim1))):
                self.x_train, self.x_test = self.x_dim2[train_index], self.x_dim2[test_index]
                self.y_train, self.y_test = self.y_dim1[train_index], self.y_dim1[test_index]

                single_fold_score = {'test': {}, 'train': {}}

                fit_start = datetime.datetime.now()
                self.model.fit(self.x_dim2, self.y_dim1)
                single_fold_score['time_taken'] = (datetime.datetime.now() - fit_start).microseconds

                # get predictions
                self.y_test_pred = self.model.predict(self.x_test)

                for k, v in self.scoring_metrics_keras_scorers.items():
                    single_fold_score['test'][k] = round(v['scorer'](y_true=self.y_test,
                                                                     y_pred=self.y_test_pred,
                                                                     **v['kwargs']), self.score_dp)

                # get predictions
                self.y_train_pred = self.model.predict(self.x_train)

                for k, v in self.scoring_metrics_keras_scorers.items():
                    single_fold_score['train'][k] = round(v['scorer'](y_true=self.y_train,
                                                                      y_pred=self.y_train_pred,
                                                                      **v['kwargs']), self.score_dp)

                all_fold_scores.append(single_fold_score)

            self.score['time_taken_total_fit_in_s'] = round(sum([k['time_taken'] for k in all_fold_scores]) / 1000000)
            self.score['time_taken_1_fold_fit_in_s'] = round(all_fold_scores[-1]['time_taken'] / 1000000)

            for j in all_fold_scores:
                j.pop('time_taken')

            self.score['test'] = {}
            self.score['train'] = {}

            for tt in all_fold_scores[-1].keys():
                for key in all_fold_scores[-1]['test'].keys():
                    self.score[tt][key + '_k_fold_mean'] = \
                        round(float(np.mean([k[tt][key] for k in all_fold_scores])), self.score_dp)
                    self.score[tt][key + '_k_fold_std'] = \
                        round(float(np.std([k[tt][key] for k in all_fold_scores])), self.score_dp)

            self.score['best_score_k_fold_mean'] = \
                round(float(np.mean([k['test'][score_criteria_for_best_model_fit] for k in all_fold_scores])), self.score_dp)
            self.score['best_score_k_fold_std'] = \
                round(float(np.std([k['test'][score_criteria_for_best_model_fit] for k in all_fold_scores])), self.score_dp)
            for key in all_fold_scores[-1]['test'].keys():
                self.score['test'][key + '_1_fold'] = all_fold_scores[-1]['test'][key]
            self.score['best_score_1_fold'] = all_fold_scores[-1]['test'][score_criteria_for_best_model_fit]

        # calculate probabilities and confusion matrix if pos
        try:
            self.y_test_pred_proba = self.model.predict_proba(self.x_test)
        except:
            self.y_test_pred_proba = None

        self.confusion_matrix: np.ndarray = make_confusion_matrix(y_true=self.y_test, y_pred=self.y_test_pred,
                                                                  labels=self.labels)

        logger.info('Training with {} done. Fit time: {}s.'.format(self.model_name,
                                                                   self.score['time_taken_total_fit_in_s']))

        self._calculate_report_for_model_keras()

    def make_roc_plot(self, pos_label: int or str):
        if self.y_test_pred_proba is None:
            raise Exception('cannot make roc plot for model {}, probability not available'.format(self.model_name))

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

        for i in ['n_classes',
                  'classification_type',
                  'model_name',
                  ]:
            report[i] = self.__getattribute__(i)

        self.report = report

    def print_report(self):
        if not self.report:
            raise Exception('Cannot print report, report has not been instantiated')
        pprint(self.report)

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

        # todo: if image side too small, this will throw and error, set image side min 30

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
                               do_grid_search: bool = False,
                               grid_search_parameters: Optional[Dict] = None,
                               score_criteria_for_best_model_fit: Optional[str] = None
                               ):

        self._init_training()

        self.model_name = model_name
        self.model = model
        logger.info('Training with {}.'.format(self.model_name))

        self.model.summary()
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=[self.scoring_metrics_tensorflow[k] for k in self.scoring_metrics_tensorflow_ordered]
        )

        sss = StratifiedShuffleSplit(n_splits=self.k_folds,
                                     test_size=self.test_size,
                                     random_state=self.random_state)

        # todo: 1 fold only here for now, add n folds
        for train_index, test_index in sss.split(x, y):

            self.x_train, self.x_test = x[train_index], x[test_index]
            self.y_train, self.y_test = y[train_index], y[test_index]
            continue

        # todo: calculate fit time, keep 4 dp
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=True,
            callbacks=callbacks
        )

        test_score = self.model.evaluate(self.x_test, self.y_test)

        self.score['test'] = {}

        for item in self.scoring_metrics_tensorflow_ordered:
            self.score['test'][item] = round(test_score[self.scoring_metrics_tensorflow_ordered.index(item)+1], self.score_dp)

        # get back list of original labels, e.g. ['cat', 'dog', etc...]
        y_test_labels = [k[0] for k in self.ohe.inverse_transform(self.y_test)]
        y_test_pred = self.model.predict(self.x_test)
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

        for i in ['n_classes',
                  'classification_type',
                  'model_name',
                  ]:
            self.report[i] = self.__getattribute__(i)

        return self.report

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
        self.score = {}
        self.model = None
        self.model_name = None
        self.best_model_parameters = {}
        self.report = {}
        self.confusion_matrix = None
        self.history = None
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        self.y_test_pred, self.y_test_pred_proba = None, None

    def _init_scoring(self):
        # todo:
        #  - change scoring technique depending wther probelm is binary or multiclass
        #  - add sample_weight to scorers

        logger.debug('returning 0 on precision score zerodivision')
        self.scoring_metrics_keras_scorers = {'accuracy_score': {'scorer': accuracy_score,
                                                                 'kwargs': {}
                                                                 },
                                              'precision_score': {'scorer': precision_score,
                                                                  'kwargs': {'average': self.score_average_method,
                                                                             'zero_division': 0}
                                                                  },
                                              'recall_score': {'scorer': recall_score,
                                                               'kwargs': {'average': self.score_average_method}
                                                               },
                                              'f1_score': {'scorer': f1_score,
                                                           'kwargs': {'average': self.score_average_method}
                                                           },
                                              'matthews_score': {'scorer': matthews_corrcoef,
                                                                 'kwargs': {}
                                                                 },
                                              # https://stackoverflow.com/questions/49061575/why-when-i-use-gridsearchcv-with-roc-auc-scoring-the-score-is-different-for-gri
                                              # 'auc_score': make_scorer(roc_auc_score, multi_class='ovr',
                                              #                         labels=self.labels,
                                              #                         greater_is_better=True,
                                              #                         needs_threshold=True
                                              #                         ),
                                              # 'zero_one_loss_score': make_scorer(zero_one_loss),
                                              }
        self.scoring_metrics_keras_scorers_for_gridsearch = {k: make_scorer(v['scorer'], **v['kwargs']) for (k, v) in
                                                             self.scoring_metrics_keras_scorers.items()}
        self.scoring_metrics_tensorflow = {'accuracy_score': 'accuracy',
                                           'precision_score': tf_precision_score(),
                                           'recall_score': tf_recall_score(),
                                           'f1_score': tf_f1_score_funk,
                                           'matthews_score': tf_matthews_score_funk,
                                           # 'auc_score': tf_auc_score()
                                           }

        self.scoring_metrics_tensorflow_ordered = ['accuracy_score',
                                                   'precision_score',
                                                   'recall_score',
                                                   'f1_score',
                                                   # 'auc_score',
                                                   'matthews_score']

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

        self.report = report
