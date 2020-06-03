from unittest import skip

from wrapml.imports.testing import *
from wrapml.imports.vanilla import pprint
from wrapml.learn.supervised.classification import ClassificationTask
from wrapml import DataGenerator


class TestClassificationTask(TestCase):

    dg = DataGenerator(n_classes=3,
                       x_shape=(100, 10))
    x, y = dg.xy()

    ct = ClassificationTask(x=x, y=y)

    ct.k_folds = 2
    ct.n_jobs = 1

    def test_report_keys(self):

        self.ct.train_with_xgboost(do_grid_search=True, k_folds=1)
        score_with_grid_search = self.ct.score

        self.ct.train_with_xgboost(do_grid_search=False, k_folds=3)
        score_without_grid_search = self.ct.score

        self.assertEqual(score_without_grid_search.keys(), score_with_grid_search.keys())
        self.assertEqual(score_without_grid_search['test'].keys(), score_with_grid_search['test'].keys())
        self.assertEqual(score_without_grid_search['train'].keys(), score_with_grid_search['train'].keys())


class TestImageGrayScaleClassificationTask(TestCase):

    dg = DataGenerator(n_classes=2,
                       x_shape=(100, 40, 40, 1))
    x, y = dg.xy()

    ct = ClassificationTask(x=x, y=y)

    ct.k_folds = 2
    ct.n_jobs = 1

    def test_knn(self):

        self.ct.train_with_knn(do_grid_search=False)
        score = self.ct.score
        accuracy = round(score['test']['accuracy_score_k_fold_mean'], 2)
        self.assertEqual(1.0, accuracy)
        report = self.ct.report
        self.ct.make_confusion_plot()
        self.ct.make_roc_plot(pos_label=self.ct.labels[0])

    def test_random_forests(self):

        self.ct.train_with_random_forests(do_grid_search=False)
        score = self.ct.score
        accuracy = round(score['test']['accuracy_score_k_fold_mean'], 2)
        self.assertEqual(1.0, accuracy)
        report = self.ct.report
        self.ct.make_confusion_plot()
        self.ct.make_roc_plot(pos_label=self.ct.labels[0])

    def test_xgboost(self):

        self.ct.train_with_xgboost(do_grid_search=False)
        score = self.ct.score
        accuracy = round(score['test']['accuracy_score_k_fold_mean'], 2)
        self.assertEqual(1.0, accuracy)
        report = self.ct.report
        self.ct.make_confusion_plot()
        self.ct.make_roc_plot(pos_label=self.ct.labels[0])

    def test_svc(self):

        self.ct.train_with_svc(do_grid_search=False)
        score = self.ct.score
        accuracy = round(score['test']['accuracy_score_k_fold_mean'], 2)
        self.assertEqual(1.0, accuracy)
        report = self.ct.report
        self.ct.make_confusion_plot()

    def test_ada(self):

        self.ct.train_with_ada(do_grid_search=False)
        score = self.ct.score
        accuracy = round(score['test']['accuracy_score_k_fold_mean'], 2)
        self.assertEqual(1.0, accuracy)
        report = self.ct.report
        self.ct.make_confusion_plot()
        self.ct.make_roc_plot(pos_label=self.ct.labels[0])

    def test_conv2d(self):
        self.ct.train_with_conv2d(epochs=10)
        report = self.ct.report


class TestScalarClassificationTask(TestCase):

    dg = DataGenerator(n_classes=2,
                       x_shape=(120, ))
    x, y = dg.xy()

    ct = ClassificationTask(x=x, y=y)

    ct.k_folds = 2
    ct.n_jobs = 1

    def test_knn(self):

        self.ct.train_with_knn()
        score = self.ct.score
        accuracy = round(score['test']['accuracy_score'], 2)
        self.assertEqual(0.58, accuracy)
        report = self.ct.report
        self.ct.make_confusion_plot()
        self.ct.make_roc_plot(pos_label=self.ct.labels[0])

    def test_knn_with_grid_search(self):

        self.ct.train_with_knn(do_grid_search=True)
        score = self.ct.score
        accuracy = round(score['test']['accuracy_score'], 2)
        self.assertEqual(0.64, accuracy)
        report = self.ct.report
        self.ct.make_confusion_plot()
        self.ct.make_roc_plot(pos_label=self.ct.labels[0])

    def test_random_forests(self):

        self.ct.train_with_random_forests()
        score = self.ct.score
        accuracy = round(score['test']['accuracy_score'], 2)
        self.assertEqual(0.6, accuracy)
        report = self.ct.report
        self.ct.make_confusion_plot()
        self.ct.make_roc_plot(pos_label=self.ct.labels[0])

    def test_xgboost(self):

        self.ct.train_with_xgboost()
        score = self.ct.score
        accuracy = round(score['test']['accuracy_score'], 2)
        self.assertEqual(0.6, accuracy)
        report = self.ct.report
        self.ct.make_confusion_plot()
        self.ct.make_roc_plot(pos_label=self.ct.labels[0])

    def test_search_estimator(self):
        self.ct.search_estimator(do_grid_search=False)
        report = self.ct.report
        self.ct.make_confusion_plot()
        self.ct.make_roc_plot(pos_label=self.ct.labels[0])

    @skip
    def test_search_estimator_with_grid_search(self):
        self.ct.search_estimator(do_grid_search=True)
        report = self.ct.report
        self.ct.make_confusion_plot()
        self.ct.make_roc_plot(pos_label=self.ct.labels[0])


class TestTimeseriesClassificationTask(TestCase):
    
    dg = DataGenerator(n_classes=2,
                       x_shape=(120, 30))
    x, y = dg.xy()

    ct = ClassificationTask(x=x, y=y)

    ct.k_folds = 2
    ct.n_jobs = 1

    # todo: coming soon, include lstm


@skip
class TestWineClassificationTask(TestCase):
    from sklearn.datasets import load_wine

    x, y = load_wine(return_X_y=True)

    def test_search_estimator_wine(self):

        ct = ClassificationTask(x=self.x, y=self.y)
        ct.n_jobs = 1
        ct.k_folds = 2

        ct.search_estimator()

        report = ct.report

        self.assertEqual('RandomForestClassifier', ct.report['best_estimator']['best_estimator_name'])

@skip
class TestIrisClassificationTask(TestCase):
    from sklearn.datasets import load_iris

    x, y = load_iris(return_X_y=True)

    def test_search_estimator_iris(self):

        ct = ClassificationTask(x=self.x, y=self.y)
        ct.n_jobs = 1
        ct.k_folds = 2

        ct.search_estimator()

        report = ct.report

        self.assertEqual('SVC', ct.report['best_estimator']['best_estimator_name'])
        ct.make_confusion_plot(normalize=True)


@skip
class TestDigitsClassificationTask(TestCase):
    from sklearn.datasets import load_digits

    x, y = load_digits(return_X_y=True)

    def test_search_estimator_digits(self):

        ct = ClassificationTask(x=self.x, y=self.y)
        ct.n_jobs = 1
        ct.k_folds = 2

        ct.search_estimator(do_grid_search=True)

        report = ct.report

        self.assertEqual('SVC', ct.report['best_estimator']['best_estimator_name'])


@skip
class TestBreastCancerClassificationTask(TestCase):
    from sklearn.datasets import load_breast_cancer

    x, y = load_breast_cancer(return_X_y=True)

    def test_search_estimator_breast_cancer(self):

        ct = ClassificationTask(x=self.x, y=self.y)
        ct.n_jobs = 1
        ct.k_folds = 2

        ct.search_estimator(do_grid_search=True)

        report = ct.report

        self.assertEqual('RandomForestClassifier', ct.report['best_estimator']['best_estimator_name'])
