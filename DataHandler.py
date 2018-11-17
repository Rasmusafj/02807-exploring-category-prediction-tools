import numpy as np
import os
import re
from random import shuffle


class DataHandler(object):

    def __init__(self,
                 directory_path="./data/dataset/",
                 test_split = 0.2,
                 balance_categories=False,
                 memory_effecient=False):

        category_files = os.listdir(directory_path)
        regex_category = "(.*)\.txt"

        self.test_split = test_split
        self.data_dict = {}
        self.index_to_category_dict = {}
        self.category_to_index = {}
        self.min_number_pages = 0

        # Loads all data into a dictionary
        if not memory_effecient:
            for i, category_file in enumerate(category_files):
                category = re.search(regex_category, category_file).group(1)
                data = []
                f = open(directory_path + category_file, 'r', encoding="utf-8")
                for line in f.readlines():
                    line = line.rstrip("\n").split(",")
                    data.append(line)

                if balance_categories and (self.min_number_pages > len(data) or self.min_number_pages == 0):
                    self.min_number_pages = len(data)

                self.data_dict[category] = data
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
            shuffle(data)
            test_nr = int(len(data) * self.test_split)
            dummy_test_X = data[:test_nr]
            self.test_X += dummy_test_X
            self.test_y += [self.category_to_index[category]] * len(dummy_test_X)

            dummy_train_X = data[test_nr:]
            self.train_X += dummy_train_X
            self.train_y += [self.category_to_index[category]] * len(dummy_train_X)

            # Reset data_dict for category
            self.data_dict[category] = []

    def get_train(self):
        return self.train_X, self.train_y

    def get_test(self):
        return self.test_X, self.test_y


if __name__ == '__main__':
    data_handler = DataHandler(balance_categories=True)
    train_X, train_y = data_handler.get_train()
    test_X, test_y = data_handler.get_test()
