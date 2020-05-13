from src.imports.science import np
from src.imports.vanilla import Dict
from src.imports.learn import pickle, StratifiedShuffleSplit
from src.imports.learn import RandomForestClassifier, KNeighborsClassifier, AdaBoostClassifier, SVC, MLPClassifier, \
    XGBClassifier
from src.imports.learn import GridSearchCV
from src.imports.learn import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef, \
    cohen_kappa_score, roc_auc_score, zero_one_loss, make_scorer
from src.utils.logging import logger
from src.constants import DEFAULT_GRID_SEARCH_PARAMETERS, CLASSIFICATION_TYPE_BINARY, CLASSIFICATION_TYPE_MULTICLASS
from src.constants import DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE


# todo HIGH PRIO:
#  - add LSTM: # https://www.analyticsvidhya.com/blog/2019/01/introduction-time-series-classification/
#  - save best model + path
#  - print report, print small report, save report + path
#  - add option print report
#  - add score given min probability for classifier which provide probability
#  - add models ranking to report
#  - add: refit best model when search all


# todo MED PRIO:
#  - linear and non linear SVC
#  - add print plots (ROC curve, confusion matrix, overfit, underfit, plot to compare models)
#  - check works with images without channels
#  - fix labels encoding
#  - class weight support
#  - add CNN tensorflow and beyond + rescale image 255 / https://www.tensorflow.org/tutorials/images/classification

# todo LOW PRIO
#  - add OneVsRestClassifier
#  - add multilabel support
#  - add cohen k-score for inter-annotators agreement
#  - dd automatic overfit / underfit detection
#  - feature selection


class TrainClassificationModel:

    def __init__(self, x: np.ndarray,
                 y: np.ndarray):

        # todo: put stuff into init

        # setup
        self.random_state = DEFAULT_RANDOM_STATE
        self.n_jobs = -1

        # scoring
        self.test_size = DEFAULT_TEST_SIZE
        self.stratify = True  # todo: at the moment we stratify by default, might offer non-stratified
        self.k_fold_cross_validation = 1
        self.fit_time_best_model_k_folds_s: int = None
        self.fit_time_total_s: int = None
        self.score: Dict = {}
        self.score_dp: int = 4
        self.score_average_method: str = 'weighted'
        self.do_grid_search: bool = False
        self.return_train_score: bool = True
        self.scoring_methods: Dict = {}
        self.score_criteria_for_best_model_fit = 'f1_score'

        # model
        self.model = None
        self.best_model_parameters: Dict = {}
        self.model_name: str = None

        self.best_parameters_from_grid_search: Dict = {}
        self.parameter_combinations_searched: int = None

        # report
        self.report: Dict = {}

        # data
        self.y_as_vector: np.ndarray = y

        if not isinstance(y, np.ndarray):
            raise Exception('y must be a numpy array')

        # check labels shape
        y_shape = y.shape

        if len(y_shape) not in (1, 2):
            raise Exception('y should be a numpy array with shape (n,) or (n, 1)')
        if len(y_shape) == 1:
            self.y_as_vector: np.ndarray = self.y_as_vector.reshape(y_shape[0], 1)

        if y_shape[0] <= 10:
            raise Exception('y should be a numpy array with shape (n,), n >= 10')

        self.n_classes: int = len(set(self.y_as_vector[:, 0]))

        if self.n_classes > 2:
            self.classification_type = CLASSIFICATION_TYPE_MULTICLASS
        elif self.n_classes == 2:
            self.classification_type = CLASSIFICATION_TYPE_BINARY
        else:
            raise Exception('n_classes must be 2 or larger')

        if not isinstance(x, np.ndarray):
            raise Exception('x must be a numpy array')

        x_shape = x.shape

        if len(x_shape) == 2:
            if x_shape[0] <= 10:
                raise Exception('If passed as vector, x should be a numpy array with shape (n, m), n >= 10')
            self.x_as_vector: np.ndarray = x
            self.x_as_image: np.ndarray = None  # todo: think about reshaping strategies
        elif len(x_shape) == 3:
            logger.warning('Be mindful, we assume x input to have shape (n_samples, x_dim, y_dim)')
            self.x_as_vector: np.ndarray = x.reshape((x_shape[0], x_shape[1] * x_shape[2]))
            self.x_as_image: np.ndarray = x.reshape((x_shape[0], 1, x_shape[1], x_shape[2]))

        elif len(x_shape) == 4:
            logger.warning('Be mindful, we assume x input to have shape (n_samples, n_channels, x_dim, y_dim)')

            self.x_as_vector: np.ndarray = x.reshape((x_shape[0], x_shape[1] * x_shape[2] * x_shape[3]))
            self.x_as_image: np.ndarray = x
        else:
            raise Exception('Shape of input x not understood, see documentation')

    def train_with_knn(self,
                       do_grid_search: bool = None,
                       grid_search_parameters: Dict = None,
                       score_criteria_for_best_model_fit: str = None,
                       **kwargs
                       ):

        model = KNeighborsClassifier(**kwargs, n_jobs=self.n_jobs)

        self._train_model_with_keras(model=model,
                                     grid_search_parameters=grid_search_parameters,
                                     score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                                     do_grid_search=do_grid_search)

    def train_with_random_forests(self,
                                  do_grid_search: bool = None,
                                  grid_search_parameters: Dict = None,
                                  score_criteria_for_best_model_fit: str = None,
                                  **kwargs):

        model = RandomForestClassifier(**kwargs, random_state=self.random_state, n_jobs=self.n_jobs)

        self._train_model_with_keras(model=model,
                                     grid_search_parameters=grid_search_parameters,
                                     score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                                     do_grid_search=do_grid_search)

    def train_with_mlp(self,
                       do_grid_search: bool = None,
                       grid_search_parameters: Dict = None,
                       score_criteria_for_best_model_fit: str = None,
                       **kwargs):

        model = MLPClassifier(**kwargs, random_state=self.random_state, early_stopping=True)

        self._train_model_with_keras(model=model,
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

        self._train_model_with_keras(model=model,
                                     grid_search_parameters=grid_search_parameters,
                                     score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                                     do_grid_search=do_grid_search)

    def train_with_ada(self,
                       do_grid_search: bool = None,
                       grid_search_parameters: Dict = None,
                       score_criteria_for_best_model_fit: str = None,
                       **kwargs):

        model = AdaBoostClassifier(**kwargs, random_state=self.random_state)

        self._train_model_with_keras(model=model,
                                     grid_search_parameters=grid_search_parameters,
                                     score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                                     do_grid_search=do_grid_search)

    def train_with_svc(self,
                       do_grid_search: bool = None,
                       grid_search_parameters: Dict = None,
                       score_criteria_for_best_model_fit: str = None,
                       **kwargs):

        model = SVC(**kwargs, random_state=self.random_state)

        self._train_model_with_keras(model=model,
                                     grid_search_parameters=grid_search_parameters,
                                     score_criteria_for_best_model_fit=score_criteria_for_best_model_fit,
                                     do_grid_search=do_grid_search)

    def train_with_lstm(self):
        # todo
        #  - https://www.curiousily.com/posts/time-series-classification-for-human-activity-recognition-with-lstms-in-keras/#classifying-human-activity
        pass

    def _train_model_with_keras(self,
                                model,
                                do_grid_search: bool,
                                grid_search_parameters: Dict,
                                score_criteria_for_best_model_fit: str,
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

        self._init_scoring()

        model_gridsearch = GridSearchCV(estimator=model,
                                        param_grid=grid_search_parameters,
                                        n_jobs=self.n_jobs,
                                        scoring=self.scoring_methods,
                                        refit=score_criteria_for_best_model_fit,
                                        cv=sss,
                                        return_train_score=True)

        model_gridsearch.fit(self.x_as_vector, self.y_as_vector)

        index_of_best_model = model_gridsearch.best_index_
        results = model_gridsearch.cv_results_

        # todo: add manual roc_auc_score here, from best model

        # let's parse score for best model, mean means mean of k-folds
        for j in ['train', 'test']:
            set_score = {}
            for k in self.scoring_methods.keys():
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
        self.best_model_parameters = self.model.get_params()

        self.fit_time_best_model_k_folds_s = round(float(model_gridsearch.refit_time_), 4)
        self.fit_time_total_s = round(float(np.sum(results['mean_fit_time'])), 4)

        logger.info('Training with {} done. Fit time: {}s.'.format(self.model_name,
                                                                   self.fit_time_total_s))

        self._calculate_report_for_model()

    def _calculate_report_for_model(self):

        report = self.score

        for i in ['fit_time_best_model_k_folds_s', 'fit_time_total_s',
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
        self.fit_time_total_s = None
        self.fit_time_best_model_k_folds_s = None
        self.score = {}
        self.model = None
        self.model_name = None
        self.best_model_parameters = {}
        self.best_parameters_from_grid_search = {}
        self.parameter_combinations_searched = None
        self.report = {}

    def _init_scoring(self):
        # todo: change scoring technique depending wther probelm is binary or multiclass
        logger.debug('returning 0 on precision score zerodivision')
        self.scoring_methods = {'accuracy_score': make_scorer(accuracy_score),
                                'f1_score': make_scorer(f1_score, average=self.score_average_method),
                                'precision_score': make_scorer(precision_score, average=self.score_average_method,
                                                               zero_division=0),
                                'recall_score': make_scorer(recall_score, average=self.score_average_method),
                                'matthews_score': make_scorer(matthews_corrcoef),
                                'zero_one_loss_score': make_scorer(zero_one_loss),
                                # 'roc_auc_score': make_scorer(roc_auc_score, average='macro', multi_class='ovo')
                                }

    def search_estimator(self, do_grid_search: bool = False):

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

    def train_all_with_search(self):
        pass
