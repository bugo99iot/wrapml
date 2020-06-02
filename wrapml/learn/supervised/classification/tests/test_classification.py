from unittest import skip

from wrapml.imports.testing import *
from wrapml.imports.vanilla import pprint
from wrapml.learn.supervised.classification import ClassificationTask
from wrapml import DataGenerator


class TestImageClassification(TestCase):

    dg = DataGenerator(n_classes=3,
                       x_shape=(60, 20, 20, 1))
    x, y = dg.xy()

    tm = ClassificationTask(x=x, y=y)

    tm.k_fold_cross_validation = 2
    tm.n_jobs = 1

    def test_knn(self):

        self.tm.train_with_knn()
        score = self.tm.score
        accuracy = round(score['test']['accuracy_score_mean'], 2)
        self.assertEqual(0.94, accuracy)
        report = self.tm.report
        small_report = self.tm.small_report
        self.tm.make_confusion_plot()

    def test_random_forests(self):

        self.tm.train_with_random_forests()
        score = self.tm.score
        accuracy = round(score['test']['accuracy_score_mean'], 2)
        self.assertEqual(0.69, accuracy)
        report = self.tm.report
        small_report = self.tm.small_report
        self.tm.make_confusion_plot()

    def test_xgboost(self):

        self.tm.train_with_xgboost()
        score = self.tm.score
        accuracy = round(score['test']['accuracy_score_mean'], 2)
        self.assertEqual(0.47, accuracy)
        report = self.tm.report
        small_report = self.tm.small_report
        self.tm.make_confusion_plot()


class TestScalarClassification(TestCase):

    dg = DataGenerator(n_classes=2,
                       x_shape=(120,))
    x, y = dg.xy()

    tm = ClassificationTask(x=x, y=y)

    tm.k_fold_cross_validation = 2
    tm.n_jobs = 1

    def test_knn(self):

        self.tm.train_with_knn()
        score = self.tm.score
        accuracy = round(score['test']['accuracy_score_mean'], 2)
        self.assertEqual(0.58, accuracy)
        report = self.tm.report
        small_report = self.tm.small_report
        self.tm.make_confusion_plot()
        self.tm.make_roc_plot(pos_label=self.tm.labels[0])

    def test_knn_with_grid_search(self):

        self.tm.train_with_knn(do_grid_search=True)
        score = self.tm.score
        accuracy = round(score['test']['accuracy_score_mean'], 2)
        self.assertEqual(0.64, accuracy)
        report = self.tm.report
        small_report = self.tm.small_report
        self.tm.make_confusion_plot()
        self.tm.make_roc_plot(pos_label=self.tm.labels[0])

    def test_random_forests(self):

        self.tm.train_with_random_forests()
        score = self.tm.score
        accuracy = round(score['test']['accuracy_score_mean'], 2)
        self.assertEqual(0.6, accuracy)
        report = self.tm.report
        small_report = self.tm.small_report
        self.tm.make_confusion_plot()
        self.tm.make_roc_plot(pos_label=self.tm.labels[0])

    def test_xgboost(self):

        self.tm.train_with_xgboost()
        score = self.tm.score
        accuracy = round(score['test']['accuracy_score_mean'], 2)
        self.assertEqual(0.6, accuracy)
        report = self.tm.report
        small_report = self.tm.small_report
        self.tm.make_confusion_plot()
        self.tm.make_roc_plot(pos_label=self.tm.labels[0])

    def test_search_estimator(self):
        self.tm.search_estimator(do_grid_search=False)
        report = self.tm.report
        small_report = self.tm.small_report
        self.tm.make_confusion_plot()
        self.tm.make_roc_plot(pos_label=self.tm.labels[0])

    @skip
    def test_search_estimator_with_grid_search(self):
        self.tm.search_estimator(do_grid_search=True)
        report = self.tm.report
        small_report = self.tm.small_report
        self.tm.make_confusion_plot()
        self.tm.make_roc_plot(pos_label=self.tm.labels[0])


@skip
class TestTrainClassificationModelWine(TestCase):
    from sklearn.datasets import load_wine

    x, y = load_wine(return_X_y=True)

    def test_search_estimator_wine(self):

        tm = ClassificationTask(x=self.x, y=self.y)
        tm.n_jobs = 1
        tm.k_fold_cross_validation = 2

        tm.search_estimator()

        self.assertEqual('RandomForestClassifier', tm.report['best_estimator']['best_estimator_name'])

@skip
class TestTrainClassificationModelIris(TestCase):
    from sklearn.datasets import load_iris

    x, y = load_iris(return_X_y=True)

    def test_search_estimator_iris(self):

        tm = ClassificationTask(x=self.x, y=self.y)
        tm.n_jobs = 1
        tm.k_fold_cross_validation = 2

        tm.search_estimator()

        self.assertEqual('SVC', tm.report['best_estimator']['best_estimator_name'])
        tm.make_confusion_plot(normalize=True)


@skip
class TestTrainClassificationModelDigits(TestCase):
    from sklearn.datasets import load_digits

    x, y = load_digits(return_X_y=True)

    def test_search_estimator_digits(self):

        tm = ClassificationTask(x=self.x, y=self.y)
        tm.n_jobs = 1
        tm.k_fold_cross_validation = 2

        tm.search_estimator(do_grid_search=True)

        self.assertEqual('SVC', tm.report['best_estimator']['best_estimator_name'])
