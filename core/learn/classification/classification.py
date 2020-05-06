from core.imports.science import np
from core.imports.vanilla import Dict, datetime
from core.imports.learn import pickle, StratifiedShuffleSplit
from core.imports.learn import RandomForestClassifier, KNeighborsClassifier, AdaBoostClassifier
from core.imports.learn import GridSearchCV
from core.imports.learn import f1_score, accuracy_score, precision_score, recall_score
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

        self.random_state = 0
        self.test_size = 0.3
        self.stratify = True  # todo: at the moment we stratify by default, might offer non-stratified
        self.k_fold_cross_validation = 1
        self.n_jobs = -1
        self.score_dp = 4

        self.training_time_per_fold_s: int = None
        self.training_time_tot_s: int = None
        self.score: Dict = {'f1_score': {}, 'precision_score': {}, 'recall_score': {}, 'accuracy_score': {}}
        self.model = None

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

    def train_with_knn(self,
                       test_size: float = None,
                       stratify: bool = None,
                       random_state: int = None,
                       k_fold_cross_validation: int = None,
                       n_jobs: int = None,
                       **kwargs
                       ):

        logger.info('Training with knn Nearest Neighbors.')

        if n_jobs is not None:
            n_jobs_for_model = n_jobs
        else:
            n_jobs_for_model = self.n_jobs

        model = KNeighborsClassifier(**kwargs, n_jobs=n_jobs_for_model)

        self._train_with_model(k_fold_cross_validation=k_fold_cross_validation,
                               test_size=test_size,
                               random_state=random_state,
                               model=model)

        logger.info('Training with knn Nearest Neighbors done. Time taken per fold: {}s.'.format(self.training_time_per_fold_s))

    def train_with_random_forests(self,
                                  test_size: float = None,
                                  stratify: bool = None,
                                  random_state: int = None,
                                  k_fold_cross_validation: int = None,
                                  n_jobs: int = None,
                                  **kwargs):

        logger.info('Training with Random Forests.')

        if n_jobs is not None:
            n_jobs_for_model = n_jobs
        else:
            n_jobs_for_model = self.n_jobs

        model = RandomForestClassifier(**kwargs, random_state=self.random_state, n_jobs=n_jobs_for_model)

        self._train_with_model(k_fold_cross_validation=k_fold_cross_validation,
                               test_size=test_size,
                               random_state=random_state,
                               model=model)

        logger.info('Training with Random Forests done. Time taken per fold: {}s.'.format(self.training_time_per_fold_s))

    def train_with_ada(self,
                       test_size: float = None,
                       random_state: int = None,
                       k_fold_cross_validation: int = None,
                       **kwargs):

        logger.info('Training with Ada Boost.')

        model = AdaBoostClassifier(**kwargs, random_state=self.random_state)

        self._train_with_model(k_fold_cross_validation=k_fold_cross_validation,
                               test_size=test_size,
                               random_state=random_state,
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
                          k_fold_cross_validation: int,
                          test_size: float,
                          random_state: int,
                          grid_search: bool = False):
        """

        :param model:
        :param k_fold_cross_validation:
        :param test_size:
        :param random_state:
        :param grid_search:

        :return:
        """

        if test_size is not None:
            test_size_for_model = test_size
        else:
            test_size_for_model = self.test_size
        if random_state is not None:
            random_state_for_model = random_state
        else:
            random_state_for_model = self.random_state
        if k_fold_cross_validation is not None:
            k_fold_cross_validation_for_model = k_fold_cross_validation
        else:
            k_fold_cross_validation_for_model = self.k_fold_cross_validation

        self.init_training()

        self.model = model

        sss = StratifiedShuffleSplit(n_splits=k_fold_cross_validation_for_model,
                                     test_size=test_size_for_model,
                                     random_state=random_state_for_model)
        logger.debug('Using StratifiedShuffleSplit for cross-validation.')

        f1_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []

        start_training = datetime.datetime.now()

        for train_index, test_index in sss.split(self.x_as_vector, self.y_as_vector):
            x_train, x_test = self.x_as_vector[train_index], self.x_as_vector[test_index]
            y_train, y_test = self.y_as_vector[train_index], self.y_as_vector[test_index]

            self.model.fit(x_train, y_train)

            y_test_pred = self.predict(x=x_test)

            score_average_method = 'weighted'
            logger.debug('using {} method as score average'.format(score_average_method))

            f1_score_fold = f1_score(y_true=y_test, y_pred=y_test_pred, average=score_average_method)
            logger.debug('returning 0 on precision score zerodivision')
            precision_score_fold = precision_score(y_true=y_test, y_pred=y_test_pred, average=score_average_method,
                                                   zero_division=0)
            recall_score_fold = recall_score(y_true=y_test, y_pred=y_test_pred, average=score_average_method)
            accuracy_score_fold = accuracy_score(y_true=y_test, y_pred=y_test_pred)

            f1_scores.append(f1_score_fold)
            precision_scores.append(precision_score_fold)
            recall_scores.append(recall_score_fold)
            accuracy_scores.append(accuracy_score_fold)

        self.score['f1_score']['folds'] = [round(float(k), self.score_dp) for k in f1_scores]
        self.score['f1_score']['avg'] = round(float(np.average(f1_scores)), self.score_dp)
        self.score['f1_score']['std'] = round(float(np.std(f1_scores)), self.score_dp)
        self.score['precision_score']['folds'] = [round(float(k), self.score_dp) for k in precision_scores]
        self.score['precision_score']['avg'] = round(float(np.average(precision_scores)), self.score_dp)
        self.score['precision_score']['std'] = round(float(np.std(precision_scores)), self.score_dp)
        self.score['recall_score']['folds'] = [round(float(k), self.score_dp) for k in recall_scores]
        self.score['recall_score']['avg'] = round(float(np.average(recall_scores)), self.score_dp)
        self.score['recall_score']['std'] = round(float(np.std(recall_scores)), self.score_dp)
        self.score['accuracy_score']['folds'] = [round(float(k), self.score_dp) for k in accuracy_scores]
        self.score['accuracy_score']['avg'] = round(float(np.average(accuracy_scores)), self.score_dp)
        self.score['accuracy_score']['std'] = round(float(np.std(accuracy_scores)), self.score_dp)

        self.training_time_tot_s = (datetime.datetime.now() - start_training).seconds
        self.training_time_per_fold_s = round(self.training_time_tot_s / k_fold_cross_validation_for_model)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def save_model(self, path: str):
        pickle.dump(self.model, open(path, 'wb'))

    def init_training(self):
        self.training_time_per_fold_s = None
        self.training_time_tot_s = None
        self.score = {'f1_score': {}, 'precision_score': {}, 'recall_score': {}, 'accuracy_score': {}}
        self.model = None

    def train_all(self):
        pass

    def train_all_with_search(self):
        pass
