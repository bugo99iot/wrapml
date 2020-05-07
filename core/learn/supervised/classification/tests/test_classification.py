from core.imports.testing import *
from core.learn.supervised.classification import TrainClassificationModel
from core.generate_data.image import ImageGenerator


class TestTrainClassificationModel(TestCase):

    ig = ImageGenerator
    ig.gen_m_x_m_grayscale(n_shots=5, image_side=15, encode_labels=False)
    x, y = ig.training_xy()

    tm = TrainClassificationModel(x=x, y=y)

    tm.k_fold_cross_validation = 2

    def test_knn(self):

        self.tm.train_with_knn()
        score = self.tm.score
        accuracy = round(score['accuracy_score']['avg'], 2)
        self.assertEqual(1.0, accuracy)

    def test_random_forests(self):

        self.tm.train_with_random_forests(do_grid_search=False)
        score = self.tm.score
        accuracy = round(score['test']['accuracy_score_mean'], 2)
        self.assertIsNotNone(accuracy)

    def test_ada(self):

        self.tm.train_with_ada(n_estimators=50, learning_rate=0.1)
        score = self.tm.score
        accuracy = round(score['accuracy_score']['avg'], 2)
        self.assertIsNotNone(accuracy)

    def test_grid_search(self):
        self.tm._grid_search_model()
