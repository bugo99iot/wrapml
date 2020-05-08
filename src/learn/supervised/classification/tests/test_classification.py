from src.imports.testing import *
from src.imports.vanilla import pprint
from src.learn.supervised.classification import TrainClassificationModel
from src.generate_data.image import ImageGenerator


class TestTrainClassificationModelMxM(TestCase):

    ig = ImageGenerator
    ig.gen_m_x_m_grayscale(n_shots=5, image_side=15, encode_labels=False)
    x, y = ig.training_xy()

    tm = TrainClassificationModel(x=x, y=y)

    tm.k_fold_cross_validation = 2

    def test_knn(self):

        self.tm.train_with_knn()
        score = self.tm.score
        accuracy = round(score['test']['accuracy_score_mean'], 2)
        self.assertEqual(1.0, accuracy)

    def test_random_forests(self):

        self.tm.train_with_random_forests(do_grid_search=False)
        score = self.tm.score
        accuracy = round(score['test']['accuracy_score_mean'], 2)
        self.assertIsNotNone(accuracy)

    def test_ada(self):

        self.tm.train_with_ada(n_estimators=50, learning_rate=0.1)
        score = self.tm.score
        accuracy = round(score['test']['accuracy_score_mean'], 2)
        self.assertIsNotNone(accuracy)


class TestTrainClassificationModelWine(TestCase):
    from sklearn.datasets import load_wine

    x, y = load_wine(return_X_y=True)

    def test_wine(self):

        tm = TrainClassificationModel(x=self.x, y=self.y)

        tm.k_fold_cross_validation = 2

        tm.search_estimator()

        self.assertEqual('RandomForestClassifier', tm.report['best_estimator']['best_estimator_name'])


class TestTrainClassificationModelIris(TestCase):
    from sklearn.datasets import load_iris

    x, y = load_iris(return_X_y=True)

    print(x.shape)

    def test_wine(self):

        tm = TrainClassificationModel(x=self.x, y=self.y)

        tm.k_fold_cross_validation = 2

        tm.search_estimator()

        self.assertEqual('SVC', tm.report['best_estimator']['best_estimator_name'])


class TestTrainClassificationModelDigits(TestCase):
    from sklearn.datasets import load_digits

    x, y = load_digits(return_X_y=True)

    def test_wine(self):

        tm = TrainClassificationModel(x=self.x, y=self.y)

        tm.k_fold_cross_validation = 2

        tm.search_estimator(do_grid_search=True)

        self.assertEqual('SVC', tm.report['best_estimator']['best_estimator_name'])