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
    wiki = WikiCorpus(in_file, lemmatize=False, dictionary=True)

    # To output pageId and title in the generator
    wiki.metadata = True

    # Token for when categories starts
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


def create_dataset():
    all_categories = ["Category:Research",
                      "Category:Library_science",
                      "Category:Culture",
                      "Category:Arts",
                      "Category:Geography",
                      "Category:Places",
                      "Category:Health",
                      "Category:Self-care",
                      "Category:History",
                      "Category:Events",
                      "Category:Formal sciences",
                      "Category:Science",
                      "Category:Natural sciences",
                      "Category:Nature",
                      "Category:People",
                      "Category:Personal_life",
                      "Category:Self",
                      #"Category:Surnames",
                      "Category:Thought",
                      "Category:Religion",
                      "Category:Belief",
                      "Category:Society",
                      "Category:Social_sciences",
                      "Category:Technology",
                      "Category:Applied_sciences"]

    base_path = "./data/categories/"
    path_cat_gen = lambda x: base_path + x.replace(":", "-") + "-pageids.txt"
    path_page_gen = lambda x: "./data/articles/" + x

    regular_expr_strip = "Category:(.*)"

    for category in all_categories:
        cat_path = path_cat_gen(category)
        f_read = open(cat_path, 'r', encoding="utf-8")
        category_name = re.search(regular_expr_strip, category).group(1)

        print("Processing: {0}".format(category_name))
        f_write = open("./data/dataset/" + category_name + ".txt", 'w', encoding="utf-8")

        all_lines = f_read.readlines()
        shuffle(all_lines)
        counter = 0
        for page_id in all_lines:
            page_id = page_id.rstrip('\n')
            filepath = path_page_gen(page_id)

            # If not a in our pre-preprocessed pages, do not include
            if not os.path.isfile(filepath):
                continue

            f_page = open(filepath, 'r', encoding="utf-8")
            f_write.write(f_page.read())
            f_page.close()
            counter += 1

            if counter % 300 == 0:
                print("Number of pages processed: {0}".format(counter))

        print("FINAL number of pages processed: {0}".format(counter))
        f_read.close()
        f_write.close()


if __name__ == '__main__':
    # Set load_only_fraction to False, if we want to generate full corpus
    # generate_corpus(load_only=10000)
    # load_categories_dict()
    # corpus_list = load_existing_corpus()
    # print(corpus_list[3][-20:])
    create_dataset()
