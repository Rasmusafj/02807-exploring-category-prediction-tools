import io
import requests


def query_gen(title):
    query = "https://en.wikipedia.org/w/api.php?action=query&list=categorymembers" \
            "&cmtitle={0}&format=json&cmlimit=max".format(title)
    return query


def extract_page_ids_for_category(category,
                                  sub_categories_queue=None,
                                  limit=10000):
    fname = "./data/categories/" + category + "-pageids.txt"
    page_ids = {}

    # If no subqueue list is given, we just start from the top
    if not sub_categories_queue:
        sub_categories_queue = [category]
        pageid_output = io.open(fname, 'w', encoding='utf-8')

    # If sub_categories_queue given, we continue from that list instead of restarting
    else:
        with open(fname, 'r', encoding='utf-8') as f:
            for pageid in f.readlines():
                page_ids[pageid.rstrip('\n')] = True

        pageid_output = io.open(fname, 'a', encoding='utf-8')

    counter = 0
    nr_requests = 0
    while counter <= limit and sub_categories_queue:
        current_category = sub_categories_queue.pop(0)
        query = query_gen(current_category)
        response = requests.get(query)
        json_data = response.json()
        nr_requests += 1
        print("number of request: {0}".format(nr_requests))

        for item in json_data["query"]["categorymembers"]:
            if "Category:" in item["title"]:
                sub_categories_queue.append(item["title"])
            else:
                pageid = str(item["pageid"])
                if not pageid in page_ids:
                    page_ids[pageid] = True
                    pageid_output.write(pageid + "\n")
                    counter += 1

            if counter > limit:
                break

    pageid_output.close()
    with io.open(get_category_fpath(category), 'w', encoding='utf-8') as f:
        [f.write(cat + "\n") for cat in sub_categories_queue]


def get_category_fpath(category):
    return "./data/categories/" + category + "-categories_in_queue.txt"


def continue_category_page_extraction(category):
    f = open(get_category_fpath(category), 'r', encoding='utf-8')
    cat_queue = [cat.rstrip('\n') for cat in f.readlines()]
    extract_page_ids_for_category(category, sub_categories_queue=cat_queue)


if __name__ == '__main__':
    # Do not loop over all categories, or risk getting banned...
    # extract_page_ids_for_category("Category:Research")
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
                      "Category:Surnames",
                      "Category:Thought",
                      "Category:Religion",
                      "Category:Belief",
                      "Category:Society",
                      "Category:Social_sciences",
                      "Category:Technology",
                      "Category:Applied_sciences"]

    # This is dummy
    current = 1
    extract_page_ids_for_category(all_categories[current])
    #continue_category_page_extraction(all_categories[current])
