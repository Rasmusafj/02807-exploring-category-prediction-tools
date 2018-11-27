import os
import re
from random import shuffle
from sklearn.feature_extraction import FeatureHasher

from utils import construct_set_similarities, construct_token_frequencies


class DataHandler(object):

    def __init__(self,
                 directory_path="./data/dataset/",
                 test_split=0.2,
                 balance_categories=False,
                 preprocessing_method=None,
                 k=100,
                 shingles_n=3,
                 debug_number=0,
                 h=2**6,
                 normalize=False):

        print("Initializing data handler...")

        category_files = os.listdir(directory_path)

        if debug_number != 0:
            shuffle(category_files)
            category_files = category_files[0:24]

        regex_category = "(.*)\.txt"

        self.test_split = test_split
        self.data_dict = {}
        self.index_to_category_dict = {}
        self.category_to_index = {}
        self.min_number_pages = debug_number
        self.preprocessing_method = preprocessing_method
        self.k = k
        self.shingles_n = shingles_n
        self.h = h
        self.normalize = normalize

        if self.preprocessing_method and preprocessing_method == "hashing_vectorize":
            self.vectorizer = FeatureHasher(n_features=self.h)  # , non_negative=True,)

        # Loads all data into a dictionary

        print("Loading data...")
        for i, category_file in enumerate(category_files):
            category = re.search(regex_category, category_file).group(1)
            print("Preprocessing category: {0}".format(category))
            data = []
            f = open(directory_path + category_file, 'r', encoding="utf-8")
            for line in f.readlines():
                line = line.rstrip("\n").split(",")
                data.append(line)

            if balance_categories and (self.min_number_pages > len(data) or self.min_number_pages == 0):
                self.min_number_pages = len(data)

            if debug_number != 0:
               data = data[:debug_number]

            self.data_dict[category] = self.preprocess(data)
            self.index_to_category_dict[i] = category
            self.category_to_index[category] = i

        if balance_categories:
            for key in self.data_dict.keys():
                self.data_dict[key] = self.data_dict[key][:self.min_number_pages]

        self.train_X = []
        self.train_y = []
        self.test_X = []
        self.test_y = []
        self.generate_data_splits()


    def get_data_dict(self):
        return self.data_dict

    def generate_data_splits(self):

        for category in self.data_dict.keys():

            data = self.data_dict[category]

            test_nr = int(len(data) * self.test_split)
            dummy_test_X = data[:test_nr]
            self.test_X += dummy_test_X
            self.test_y += [self.category_to_index[category]] * len(dummy_test_X)

            dummy_train_X = data[test_nr:]
            self.train_X += dummy_train_X
            self.train_y += [self.category_to_index[category]] * len(dummy_train_X)

            # Reset data_dict for category
            self.data_dict[category] = []

    def preprocess(self, data):
        if not self.preprocessing_method:
            return data

        elif self.preprocessing_method == "min_hash_signatures":
            return construct_set_similarities(data, self.k, self.shingles_n, method="permutation")

        elif self.preprocessing_method == "hashing_vectorize":
            return construct_token_frequencies(data, self.vectorizer, self.normalize)

        else:
            print("Preprocessing method not supported was given.")
            return data

    def get_train(self):
        return self.train_X, self.train_y

    def get_test(self):
        return self.test_X, self.test_y


if __name__ == '__main__':
    data_handler = DataHandler(balance_categories=True,
                               preprocessing_method="hashing_vectorize",
                               debug_number=500)
    train_X, train_y = data_handler.get_train()
    test_X, test_y = data_handler.get_test()
