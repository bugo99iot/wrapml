from wrapml.imports.vanilla import logger, Tuple, List
from wrapml.imports.science import np, random
from wrapml.data_generator.utils import get_n_names


class DataGenerator:

    def __init__(self,
                 n_classes: int,
                 x_shape: Tuple or int,
                 label_type: str = 'str',
                 y_dim: int = 2,
                 random_state: int = 0):

        self.n_classes: int = n_classes
        if isinstance(int, type(x_shape)):
            x_shape = (x_shape,)
        self.x_shape: Tuple = x_shape
        self.x_dim = len(self.x_shape)
        self.label_type: int or str = label_type
        if self.x_dim not in (1, 2, 3, 4):
            raise Exception('x_shape must have dim 1, 2, 3 or 4')

        self.n_samples: int = self.x_shape[0]

        if self.n_samples < 10:
            raise Exception('x_shape first dimension (n_samples) must be 1o or larger')

        if self.label_type not in ('int', 'str'):
            raise Exception("label_type must be 'int' or 'str'")

        self.y_dim = y_dim
        if self.y_dim not in (1, 2):
            raise Exception('y_dim must be 1 or 2')

        self.random_state: int = random_state

        self.labels: List = [j for j in range(self.n_classes)] if self.label_type == 'int' \
            else get_n_names(n=self.n_classes, random_state=self.random_state)

        np.random.seed(random_state)

    def xy(self) -> Tuple[np.ndarray, np.ndarray]:

        y = [random.choice(self.labels) for i in range(self.n_samples)]
        x = []
        x_shape_exclude_n_samples = tuple(list(self.x_shape)[1:])

        for label in y:
            label_index = self.labels.index(label)

            # mean and standard deviation
            mu = 30.0*(label_index+1)
            sigma = 30.0

            x.append(np.random.normal(mu, sigma, x_shape_exclude_n_samples).tolist())

        # use label position to shift data or to add zeros

        x = np.array(x)
        y = np.array(y)

        if self.x_dim == 2:
            y = y.reshape(y.shape[0], 1)

        if self.label_type == 'int':
            y = y.astype('int32')

        if x.shape != self.x_shape:
            raise Exception('could not reproduce x_shape')

        return x, y
