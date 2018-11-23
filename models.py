"""
All models for this project should be derived from the abstract AbstractModel class. This is
to make sure that they all implement the same functions.

General code should be put in the abstract class.
"""
import numpy as np
from abc import ABC, abstractmethod
from DataHandler import DataHandler
from utils import jaccard_distance
import mmh3
from sklearn import svm


class AbstractModel(ABC):
    """
    Base class for all implemented models.
    """

    def __init__(self, **kwargs):

        self.data_handler = DataHandler(balance_categories=True, **kwargs)

        self.train_X = self.data_handler.train_X
        self.train_y = self.data_handler.train_y
        self.test_X = self.data_handler.test_X
        self.test_y = self.data_handler.test_y

        self.not_implemented_string = "Not implemented..."

    @abstractmethod
    def fit_data(self):
        """
        Call fit_data to fit data to model

        :return:
        """
        pass

    @abstractmethod
    def predict(self, document):
        """
        Predicts a single document
        :param document: Document to be classified
        :return: Category prediction index
        """
        pass

    def evaluate_on_test(self):
        """
        Evaluates test accuracy
        """
        print("Evaluating accuracy on test...")
        predictions = []
        for i in range(len(self.test_X)):
            predictions.append(self.predict(self.test_X[i]))
        accuracy = np.sum((np.asarray(predictions) - self.test_y) == 0) / len(self.test_X)
        print("Accuracy for test set: {0}".format(accuracy))
        return accuracy


    def predict_new(self, documents):
        """
        :documents: List of documents to be predicted
        :return: Category prediction
        """
        predictions = []
        documents = self.preprocess_documents(documents)
        for document in documents:
            prediction = self.predict(document)
            predictions.append(self.data_handler.index_to_category_dict[prediction])

        return predictions

    def preprocess_documents(self, documents):
        """
        :param documents: List of documents to be preprocessed
        :return: preprocessed documents
        """
        return self.data_handler.preprocess(documents)


class SetSimiliaritiesKNN(AbstractModel):

    def __init__(self,
                 k_neighbours=5,
                 k_hash_functions=100,
                 n_shingles=2,
                 **kwargs):

        print("Using Set Similarities KNN...")
        self.k_neighbours = k_neighbours

        args_dict = kwargs
        args_dict["preprocessing_method"] = "min_hash_signatures"
        args_dict["k"] = k_hash_functions
        args_dict["shingles_n"] = n_shingles

        super().__init__(**args_dict)
        self.fit_data()

    def fit_data(self):
        # Construct signature matrix in numpy and also convert test to numpy array
        self.train_X = np.asarray(self.train_X)
        self.train_y = np.asarray(self.train_y)
        self.test_X = np.asarray(self.test_X)
        self.test_y = np.asarray(self.test_y)

    def predict(self, x):
        # Get closes k neighbours

        # First element is the largest distance
        closest_k_neighbours = [0] * self.k_neighbours
        closest_k_neighbours_categories = [-1] * self.k_neighbours

        for i in range(len(self.train_X)):
            distance = jaccard_distance(x, self.train_X[i])
            for j in range(self.k_neighbours - 1):
                if distance > closest_k_neighbours[j + 1]:
                    if j == self.k_neighbours - 2:
                        closest_k_neighbours.insert(j+2, distance)
                        closest_k_neighbours.pop(0)

                        closest_k_neighbours_categories.insert(j+2, self.train_y[i])
                        closest_k_neighbours_categories.pop(0)

                    continue

                if distance > closest_k_neighbours[j]:
                    closest_k_neighbours.insert(j+1, distance)
                    closest_k_neighbours.pop(0)

                    closest_k_neighbours_categories.insert(j+1, self.train_y[i])
                    closest_k_neighbours_categories.pop(0)
                    break
                else:
                    break

        # Prediction is the maximum amount of closest categories
        # If tie, pick first
        return max(set(closest_k_neighbours_categories), key=closest_k_neighbours_categories.count)


class LSHMinHash(AbstractModel):

    def __init__(self,
                 k_neighbours=5,
                 k_hash_functions=100,
                 n_shingles=2,
                 bands=25,
                 **kwargs):

        print("Using MinHash Local Sensitivity Hashing model...")

        self.k_neighbours = k_neighbours
        self.bands = bands

        args_dict = kwargs
        args_dict["preprocessing_method"] = "min_hash_signatures"
        args_dict["k"] = k_hash_functions
        args_dict["shingles_n"] = n_shingles

        super().__init__(**args_dict)
        self.fit_data()

    def fit_data(self):

        self.train_X = np.asarray(self.train_X)
        self.train_y = np.asarray(self.train_y)

        # Make sure bands is fine
        while self.data_handler.k % self.bands != 0:
            self.bands += 1

        self.r = int(self.data_handler.k / self.bands)

        print("Using bands: {0}, rows: {1}".format(self.bands, self.r))

        self.bands_dict = {}

        # Find all indices for X where first
        for i in range(len(self.train_X)):
            for j in range(0, self.data_handler.k, self.r):
                hashed_value = mmh3.hash(self.train_X[i][j:j+self.r], j)
                if hashed_value not in self.bands_dict:
                    self.bands_dict[hashed_value] = [i]
                else:
                    self.bands_dict[hashed_value].append(i)

        self.test_X = np.asarray(self.test_X)
        self.test_y = np.asarray(self.test_y)


    def predict(self, x):
        # Get closes k neighbours

        # First element is the largest distance
        closest_k_neighbours = [0] * self.k_neighbours
        closest_k_neighbours_categories = [-1] * self.k_neighbours

        indices_to_try = []
        for j in range(0, self.data_handler.k, self.r):
            hashed_value = mmh3.hash(x[j:j + self.r], j)
            if hashed_value in self.bands_dict:
                indices_to_try += self.bands_dict[hashed_value]

        indices_to_try = set(indices_to_try)
        for i in indices_to_try:
            distance = jaccard_distance(x, self.train_X[i])
            for j in range(self.k_neighbours - 1):
                if distance > closest_k_neighbours[j + 1]:
                    if j == self.k_neighbours - 2:
                        closest_k_neighbours.insert(j + 2, distance)
                        closest_k_neighbours.pop(0)

                        closest_k_neighbours_categories.insert(j + 2, self.train_y[i])
                        closest_k_neighbours_categories.pop(0)

                    continue

                if distance > closest_k_neighbours[j]:
                    closest_k_neighbours.insert(j + 1, distance)
                    closest_k_neighbours.pop(0)

                    closest_k_neighbours_categories.insert(j + 1, self.train_y[i])
                    closest_k_neighbours_categories.pop(0)
                    break
                else:
                    break
        # Prediction is the maximum amount of closest categories
        # If tie, pick first
        return max(set(closest_k_neighbours_categories), key=closest_k_neighbours_categories.count)


class DummyMachineLearningModel(AbstractModel):
    def __init__(self,
                 C_id=1.0,
                 kernel_id='rbf',
                 degree_id=3,
                 gamma_id='auto',
                 **kwargs):

        print("Using Support Vector Classification...")

        self.C_id = C_id
        self.kernel_id = kernel_id
        self.degree_id = degree_id
        self.gamma_id = gamma_id

        # The following must be called
        args_dict = kwargs
        args_dict["preprocessing_method"] = "hashing_vectorize"
        super().__init__(**args_dict)
        self.fit_data()

    def fit_data(self):

        # NOTE: could use CV to choose optimal kernel function.

        self.train_X = np.asarray(self.train_X)
        self.train_y = np.asarray(self.train_y)

        model = svm.SVC(C=self.C_id,
                        kernel=self.kernel_id,
                        degree=self.degree_id,
                        gamma=self.gamma_id)
        self.model = model.fit(self.train_X, self.train_y)


    def predict(self, x):

        self.test_X = np.asarray(self.test_X)
        #self.test_y = np.asarray(self.test_y)

        return self.model.predict(self.test_X)


if __name__ == '__main__':
    # Testing

    # KNNSimilarities no LSH
    """
    arguments = {
        "k_neighbours": 10,
        "k_hash_functions": 300,
        "n_shingles": 2,
    }
    # Pipeline for the AbstractModel implementation
    model = SetSimiliaritiesKNN(**arguments)
    accuracy = model.evaluate_on_test()
    print("Accuracy is: {0}".format(accuracy))
    """
    arguments = {
        "k_neighbours": 3,
        "k_hash_functions": 100,
        "n_shingles": 1,
        "bands": 50,
        "debug_number": 0
    }

    # Pipeline for the AbstractModel implementation
    model = LSHMinHash(**arguments)
    accuracy = model.evaluate_on_test()
    print("Accuracy is: {0}".format(accuracy))
