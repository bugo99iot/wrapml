from core.imports.testing import *
from core.imports.science import np
from core.generate_data.utils.utils import get_n_names


class TestImageGenerator(TestCase):

    def test_get_n_names(self):

        self.assertEqual(['genevieve', 'patricia', 'theresa', 'pearle', 'gail'], get_n_names(n=5))
