from wrapml.imports.science import np
from wrapml.imports.vanilla import logger, Tuple
from wrapml.generate_data.utils.utils import get_n_names
from wrapml.utils.utils import DataProcessor
from wrapml.constants import DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE


class ImageGenerator(DataProcessor):

    x: np.ndarray = None
    y: np.ndarray = None
    y_categorical: np.ndarray = None
    x_train: np.ndarray = None
    y_train: np.ndarray = None
    x_test: np.ndarray = None
    y_test: np.ndarray = None

    @classmethod
    def gen_m_x_m_grayscale(cls, n_shots: int,
                            image_side: int,
                            rescale: bool = False,
                            encode_labels: bool = True,
                            test_size: float = DEFAULT_TEST_SIZE,
                            random_state: int = DEFAULT_RANDOM_STATE,
                            stratify: bool = True):
        """
        Generate dummy data, squared images with 1 pixel of value 1.0 and all other pixels of value 0.0

        :param n_shots:
        :param image_side:
        :param rescale:
        :param encode_labels:
        :param test_size:
        :param random_state:
        :param stratify:
        :return:
        """

        n_categories = image_side ** 2
        categories_names = get_n_names(n=n_categories)
        n_images = n_shots * n_categories
        # 1 stands for grayscale
        data = np.empty(shape=(n_images, 1, image_side, image_side), dtype='float64')
        labels = []
        image_empty = np.zeros(shape=(image_side, image_side))
        k = 0
        l = 0
        for i in range(image_side):
            for j in range(image_side):
                image = image_empty.copy()
                image[i, j] = 1.0
                for ns in range(n_shots):
                    data[k, 0] = image
                    labels.append(categories_names[l])
                    k += 1
                l += 1
        if rescale:
            data = data / 255.0

        cls.x = data
        cls.y = np.array(labels)

        if encode_labels:
            cls.y_categorical = cls.y
            cls.y = cls.y_categorical_to_encoded(y=cls.y)

        cls.x_train, cls.x_test, cls.y_train, cls.y_test = cls.train_test_split(x=cls.x,
                                                                                y=cls.y,
                                                                                test_size=test_size,
                                                                                stratify=stratify,
                                                                                random_state=random_state,
                                                                                )

        logger.info('Generated {} images given {} categories of {} shots each.'.format(n_images, n_categories, n_shots))
        # logger.info('X shape: {}'.format(cls.x.shape))
        # logger.info('Y shape: {}'.format(cls.y.shape))
        logger.info('x_train shape: {}'.format(cls.x_train.shape))
        logger.info('y_train shape: {}'.format(cls.y_train.shape))
        logger.info('x_test shape: {}'.format(cls.x_test.shape))
        logger.info('y_test shape: {}'.format(cls.y_test.shape))

    @classmethod
    def training_xy(cls) -> Tuple:
        return cls.x, cls.y

    @classmethod
    def training_tuple(cls) -> Tuple:
        logger.info('Returning training tuple x_train, x_test, y_train, y_test')
        return cls.x_train, cls.x_test, cls.y_train, cls.y_test
