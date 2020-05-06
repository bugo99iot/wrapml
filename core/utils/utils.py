from core.imports.science import *
from core.imports.vanilla import *
from core.imports.learn import *
from core.imports.learn import train_test_split


class DataProcessor:

    le = LabelEncoder()

    @classmethod
    def train_test_split(cls,
                         x: np.ndarray,
                         y: np.ndarray,
                         test_size: float = 0.3,
                         stratify: bool = True,
                         hard_stratify: bool = True,
                         random_state: int = 0) -> Tuple:

        # todo: hard stratify

        if stratify:
            stratify_item = y
        else:
            stratify_item = None

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                            random_state=random_state,
                                                            stratify=stratify_item)

        return x_train, x_test, y_train, y_test

    @classmethod
    def y_categorical_to_encoded(cls, y: np.ndarray) -> np.ndarray:

        y_numerical = cls.le.fit_transform(y=y)
        return y_numerical

    @classmethod
    def y_encoded_to_categorical(cls, y: np.ndarray) -> np.ndarray:

        return cls.le.inverse_transform(y)