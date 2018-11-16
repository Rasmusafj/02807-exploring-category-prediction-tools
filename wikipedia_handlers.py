"""
Steps needed to run functions from this file:

- Download wiki dump from https://dumps.wikimedia.org/enwiki/latest/
    - Tested version: enwiki-latest-pages-articles.xml.bz2
    - Place it in a ./data/ dir or specify in_file to function
"""
import re
from random import shuffle

from gensim.corpora import WikiCorpus
import io
import json

import os.path


def generate_corpus(load_only=None,
                    in_file="./data/enwiki-latest-pages-articles.xml.bz2"):
    """
    Loads the wikidump into a txt file that can easily be processed
    Takes approximately 8hours to run on 2010 dump, so it will take loooong...

    :param load_only: Breaks after load_only number of articles if set
    :param in_file: dump file
    :param out_file: The txt file where the extracted corpus is saved
    :return:
    """

    print('Extracting corpus from dump file...')

    # Default tokenizing parameters for WikiCorpus
    # article_min_tokens = 50
    # token_min_len = 2
    # token_max_len = 15
    # We should consider using max_token = 30ish (ask TA?)
    wiki = WikiCorpus(in_file, lemmatize=False, dictionary={})

    # To output pageId and title in the generator
    wiki.metadata = True

    # Token for when categories starts
    categories_dict = {}
    counter = 0

    for (tokens, (pageid, title)) in wiki.get_texts():
        fname = "./data/articles/" + pageid

        # If file already create continue
        if os.path.isfile(fname):
            continue

        output = io.open(fname, 'w', encoding='utf-8')
        output.write(bytes(','.join(tokens), 'utf-8').decode('utf-8') + '\n')
        output.close()

        # Monitor process
        counter = counter + 1
        if counter % 1000 == 0:
            print('Processed ' + str(counter) + ' articles')
            if load_only and counter % load_only == 0:
                break

    print('Processed ' + str(counter) + ' articles')
    with open('./data/categories_final.txt', 'w') as file:
        file.write(json.dumps(categories_dict))


def load_existing_corpus(in_file="./data/wiki_en_with_categories.txt"):
    """
    Loads the first 10000 wikipedia articles into a list.
    Care for memory explode using full txt-file.
    ToDo: Implement generators for data_handling.

    :param in_file: Txt file generated from generate_corpus() method
    :return: List where each element is a tokenized wiki article
    """

    print('Loading corpus from txt file...')
    corpus_list = []
    file_reader = io.open(in_file, encoding='utf-8')
    for i in range(100):
        corpus_list.append(file_reader.readline())
    return corpus_list


def data_generator():
    batch_size = 1000
    articles = os.listdir("data/articles/")
    shuffle(articles)
    counter = 0

    while True:
        data = []
        for fname in articles[counter:counter+batch_size]:

            with open(fname, 'r') as f:
                data.append(",".split(f.read()))

            if counter % len(articles) == 0:
                counter = 0

        counter += batch_size
        yield data


def load_categories_dict(in_filepath="./data/categories_final.txt"):
    with open(in_filepath, 'r') as f:
        category_dict = json.load(f)
    return category_dict


if __name__ == '__main__':
    # Set load_only_fraction to False, if we want to generate full corpus
    generate_corpus(load_only=10000)
    # load_categories_dict()
    # corpus_list = load_existing_corpus()
    # print(corpus_list[3][-20:])
