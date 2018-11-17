"""
All models for this project should be derived from the abstract AbstractModel class. This is
to make sure that they all implement the same functions.

General code should be put in the abstract class.
"""

from abc import ABC, abstractmethod
from DataHandler import DataHandler

class AbstractModel(ABC):
    """
    Base class for all implemented models.
    """

    def __init__(self, datahandler_type=None, **kwargs):

        self.data_handler = DataHandler(balance_categories=True,
                                        preprocessing_method=datahandler_type)


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

    def __init__(self, **kwargs):
        super().__init__(datahandler_type="similarities", **kwargs)

    def preprocess_data(self):
        print(self.not_implemented_string)

    def evaluate(self):
        print(self.not_implemented_string)


class DummyMachineLearningModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(datahandler_type="bag_of_words", **kwargs)

    def preprocess_data(self):
        print(self.not_implemented_string)

    def evaluate(self):
        print(self.not_implemented_string)
