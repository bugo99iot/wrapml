

class ModelNotTrainableException(Exception):
    """
    Exception raised when model cannot be trained given shape of x.
    """

    def __init__(self, message: str or None = None, extra: dict = None):
        self.message = message
        self.extra = extra

    def __str__(self):
        return str(self.message)
