"""
All models for this project should be derived from the abstract AbstractModel class. This is
to make sure that they all implement the same functions.

General code should be put in the abstract class.
"""

from abc import ABC, abstractmethod


class AbstractModel(ABC):
    """
    Base class for all implemented models.
    """

    def __init__(self):
        self.not_implemented_string = "Not implemented..."

    def init_data(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

    @abstractmethod
    def preprocess_data(self):
        """
        All classes implemented preprocess_data should call super.init_data() after preprocessing is complete
        """
        pass

    @abstractmethod
    def evaluate(self):
        pass


class SetSimiliaritiesKNN(AbstractModel):

    def __init__(self, **args):
        super().__init__()

    def preprocess_data(self):
        print(self.not_implemented_string)

    def evaluate(self):
        print(self.not_implemented_string)


class DummyMachineLearningModel(AbstractModel):
    def __init__(self, **args):
        super().__init__()

    def preprocess_data(self):
        print(self.not_implemented_string)

    def evaluate(self):
        print(self.not_implemented_string)
