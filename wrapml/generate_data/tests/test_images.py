from wrapml.imports.testing import TestCase
from wrapml.imports.science import np
from wrapml.generate_data.image import ImageGenerator


class TestImageGenerator(TestCase):

    def test_basic_functionality(self):
        ig = ImageGenerator
        ig.gen_m_x_m_grayscale(n_shots=4, image_side=2, encode_labels=False, test_size=0.33)
        x_train, x_test, y_train, y_test = ig.training_tuple()
        x, y = ig.training_xy()

        y_test_expected = np.array(['patricia', 'pearle', 'patricia', 'theresa', 'genevieve', 'genevieve'])

        np.testing.assert_array_equal(y_test_expected, y_test)
