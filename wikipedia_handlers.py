"""
Steps needed to run functions from this file:

- Download wiki dump from https://dumps.wikimedia.org/enwiki/latest/
    - Tested version: enwiki-latest-pages-articles.xml.bz2
    - Place it in a ./data/ dir or specify in_file to function

To get data for categories, we need to manually remove from 'categories'
from IGNORED_NAMESPACES in gensim.corpoora.wikicorpus
"""

from gensim.corpora import WikiCorpus
import io


def generate_corpus(load_only_fraction=True,
                    in_file="./data/enwiki-latest-pages-articles.xml.bz2",
                    out_file="./data/wiki_en.txt"):
    """
    Loads the wikidump into a txt file that can easily be processed
    Takes approximately 8hours to run on 2010 dump, so it will take loooong...

    :param load_only_fraction: Breaks after 10000 articles if True
    :param in_file: dump file
    :param out_file: The txt file where the extracted corpus is saved
    :return:
    """

    print('Extracting corpus from dump file...')
    output = io.open(out_file, 'w', encoding='utf-8')

    # Default tokenizing parameters for WikiCorpus
    # article_min_tokens = 50
    # token_min_len = 2
    # token_max_len = 15
    # We should consider using max_token = 30ish (ask TA?)
    wiki = WikiCorpus(in_file, lemmatize=False, dictionary={})

    # To output pageId and title in the generator
    wiki.metadata = True

    # Define two unique tokens, so we easily can extract page id and article name
    title_seperator = "::title::"
    page_id_seperator = "::pageid::"

    counter = 0
    for (tokens, (pageid, title)) in wiki.get_texts():
        output.write(title + title_seperator + pageid + page_id_seperator + bytes(','.join(tokens), 'utf-8').decode('utf-8') + '\n')

        # Monitor process
        counter = counter + 1
        if counter % 10000 == 0:
            print('Processed ' + str(counter) + ' articles')
            if load_only_fraction:
                break

    output.close()


def load_existing_corpus(in_file="./data/wiki_en.txt"):
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


if __name__ == '__main__':
    # Set load_only_fraction to False, if we want to generate full corpus
    generate_corpus(load_only_fraction=True)
    # corpus_list = load_existing_corpus()
    # print(corpus_list[3][-20:])
