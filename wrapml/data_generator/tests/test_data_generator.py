from wrapml.imports.testing import TestCase
from wrapml.imports.science import np

from wrapml import DataGenerator


class TestDataGenerator(TestCase):

    def test_it(self):
        x_shape_expected = (60, 10, 10, 1)
        dg = DataGenerator(n_classes=3,
                           x_shape=x_shape_expected)

        x, y = dg.xy()

        self.assertEqual(x_shape_expected, x.shape)
