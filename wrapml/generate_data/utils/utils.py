import names as pynames
from wrapml.imports.vanilla import List
import random


def get_n_names(n: int, random_state: int = 0) -> List[str]:
    # get a list of n unique lower-cased first names, e.g. ['nina', 'james', ...]
    random.seed(random_state)
    names = []
    while True:
        new_name = pynames.get_first_name().lower()
        if new_name not in names:
            names.append(new_name)
        if len(names) == n:
            return names
