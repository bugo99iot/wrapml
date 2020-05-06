from core.imports.science import np
from core.imports.vanilla import Dict, datetime
from core.imports.learn import pickle, StratifiedShuffleSplit
from core.imports.learn import RandomForestClassifier, KNeighborsClassifier, AdaBoostClassifier, SVC
from core.imports.learn import GridSearchCV
from core.imports.learn import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from core.utils.logging import logger


# todo: - add parameter search
#  - add train all with search
#  - add print plots
#  - check works with vecotors
#  - check works with images without channels
#  - fix labes encoding
#  - class weight support
#  - add MLP from tensorflow
#  - add cov2d, cov1d, timeseries correlation models
#  - add probability
#  - add score given min probability for classifier which provide probability


class TrainClassificationModel:

    def __init__(self, x: np.ndarray,
                 y: np.ndarray):

        # todo: put stuff into init

        self.random_state = 0
        self.test_size = 0.3
        self.stratify = True  # todo: at the moment we stratify by default, might offer non-stratified
        self.k_fold_cross_validation = 1
        self.n_jobs = -1

        self.training_time_per_fold_s: int = None
        self.training_time_tot_s: int = None
        self.score: Dict = {}
        self.score_dp = 4
        self.best_param = None
        self.model = None

        self.score_average_method = 'weighted'
        self.do_grid_search = False

        self.y_as_vector = y

        if not isinstance(y, np.ndarray):
            raise Exception('y must be a numpy array')

        # check labels shape
        y_shape = y.shape

        if len(y_shape) != 1:
            raise Exception('y should be a numpy array with shape (n,)')

        if y_shape[0] <= 10:
            raise Exception('y should be a numpy array with shape (n,), n >= 10')

        if not isinstance(x, np.ndarray):
            raise Exception('x must be a numpy array')

        x_shape = x.shape

        if len(x_shape) == 1:
            if x_shape[0] <= 10:
                raise Exception('if passed as vector, x should be a numpy array with shape (n,), n >= 10')
            self.x_as_vector = x
            self.x_as_image = None  # todo: think about reshaping strategies
        elif len(x_shape) == 3:
            logger.warning('be mindful, we assume x input to have shape (n_samples, x_dim, y_dim)')
            self.x_as_vector = x.reshape((x_shape[0], x_shape[1] * x_shape[2]))
            self.x_as_image = x.reshape((x_shape[0], 1, x_shape[1], x_shape[2]))

        elif len(x_shape) == 4:
            logger.warning('be mindful, we assume x input to have shape (n_samples, n_channels, x_dim, y_dim)')

            self.x_as_vector = x.reshape((x_shape[0], x_shape[1] * x_shape[2] * x_shape[3]))
            self.x_as_image = x
        else:
            raise Exception('shape of input x not understood, see documentation')

        self.scoring = None
        self.score_criteria_for_best_model_fit = 'f1_score'

    def train_with_knn(self,
                       **kwargs
                       ):

        logger.info('Training with knn Nearest Neighbors.')

        model = KNeighborsClassifier(**kwargs, n_jobs=self.n_jobs)

        self._train_with_model(k_fold_cross_validation=k_fold_cross_validation,
                               model=model)

        logger.info('Training with knn Nearest Neighbors done. Time taken per fold: {}s.'.format(self.training_time_per_fold_s))

    def train_with_random_forests(self,
                                  do_grid_search: bool = None,
                                  search_parameters: Dict = None,
                                  score_criteria_for_best_model_fit: str = None,
                                  **kwargs):

        logger.info('Training with Random Forests.')

        if do_grid_search is not None:
            do_grid_search_for_model = do_grid_search
        else:
            do_grid_search_for_model = self.do_grid_search

        model = RandomForestClassifier(**kwargs, random_state=self.random_state, n_jobs=self.n_jobs)

        if not do_grid_search_for_model:
            search_parameters = {}
        else:
            if search_parameters:
                pass
            else:
                search_parameters = {'n_estimators': [10, 50, 100, 200],
                                     'class_weight': ['balanced', None],
                                     'min_samples_split': [0.2, 0.8]}

        score_criteria_for_best_model_fit = score_criteria_for_best_model_fit \
            if score_criteria_for_best_model_fit else self.score_criteria_for_best_model_fit

        self._train_with_model(model=model,
                               search_parameters=search_parameters,
                               score_criteria_for_best_model_fit=score_criteria_for_best_model_fit)

        # todo: fix whether search is going on or not
        logger.info('Training with Random Forests done. Time taken per fold: {}s.'.format(self.training_time_per_fold_s))

    def train_with_ada(self,
                       **kwargs):

        logger.info('Training with Ada Boost.')

        model = AdaBoostClassifier(**kwargs, random_state=self.random_state)

        self._train_with_model(k_fold_cross_validation=self.k_fold_cross_validation,
                               model=model)

        logger.info('Training with Ada Boost done. Time taken per fold: {}s.'.format(self.training_time_per_fold_s))

    def train_with_svc(self,
                       test_size: float = None,
                       random_state: int = None,
                       k_fold_cross_validation: int = None,
                       **kwargs):
        return

    def _train_with_model(self,
                          model,
                          search_parameters: Dict,
                          score_criteria_for_best_model_fit: str):
        """

        :param model:
        :param search_parameters:

        :return:
        """

        self._init_training()

        logger.debug('Using StratifiedShuffleSplit for cross-validation.')
        sss = StratifiedShuffleSplit(n_splits=self.k_fold_cross_validation,
                                     test_size=self.test_size,
                                     random_state=self.random_state)

        self._init_scoring()

        model_gridsearch = GridSearchCV(estimator=model,
                                        param_grid=search_parameters,
                                        n_jobs=self.n_jobs,
                                        scoring=self.scoring,
                                        refit=score_criteria_for_best_model_fit,
                                        cv=sss,
                                        return_train_score=True)

        model_gridsearch.fit(self.x_as_vector, self.y_as_vector)

        index_of_best_model = model_gridsearch.best_index_
        results = model_gridsearch.cv_results_

        # let's parse score for best model, mean means mean of k-folds
        for j in ['train', 'test']:
            set_score = {}
            for k in self.scoring.keys():
                set_score[k + '_mean'] = round(float(results['mean_' + j + '_' + k][index_of_best_model]), self.score_dp)
                set_score[k + '_std'] = round(float(results['std_' + j + '_' + k][index_of_best_model]), self.score_dp)
            self.score[j] = set_score

        self.score['best_score_criteria'] = score_criteria_for_best_model_fit
        self.score['best_score'] = self.score['test'][score_criteria_for_best_model_fit + '_mean']
        if round(float(model_gridsearch.best_score_), self.score_dp) != self.score['best_score']:
            raise Exception('could not identify best score')

        self.best_param = model_gridsearch.best_params_
        self.model = model_gridsearch.best_estimator_

        # todo: time taken? best param?

    def predict(self, x: np.ndarray):
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray):
        return self.model.predict_proba(x)

    def save_model(self, path: str):
        pickle.dump(self.model, open(path, 'wb'))

    def _init_training(self):
        self.training_time_per_fold_s = None
        self.training_time_tot_s = None
        self.score = {}
        self.model = None

    def _init_scoring(self):
        logger.debug('returning 0 on precision score zerodivision')
        self.scoring = {'accuracy_score': make_scorer(accuracy_score),
                        'f1_score': make_scorer(f1_score, average=self.score_average_method),
                        'precision_score': make_scorer(precision_score, average=self.score_average_method,
                                                       zero_division=0),
                        'recall_score': make_scorer(recall_score, average=self.score_average_method)}

    def train_all(self):
        pass

    def train_all_with_search(self):
        pass
