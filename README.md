# 02807-exploring-category-prediction-tools
A universal category prediction tool based on Wikipedia created for the 
DTU course 02807 - Computational Tools for Data science.

The project was created by Sabine Fie Hansen (s134583) and Rasmus Arpe Fogh Jensen (s134843). 

## Code guide
Since the code is not run by simply calling a main method, a guide has been made.
To see which external python libraries are needed to run the code, we refer to `requirements.txt`
or `environment.yml`. To setup a virtual environment using anaconda run 


`conda env create -f environment.yml` 

which will create the virtual environment 02807-project, which can be activated by

`source activate 02807-project`

To setup a virtual environment using pip first create the virtual environment with
python 3.7 and then run

`pip install -r requirements.txt`

Then you may activate the created environment.

### Creating the dataset
The dataset files are included in the repository, so preprocessing is not
neccesary in order to test the models or run the category prediction app.
However, if you wish to create the dataset using the same approach, the process
can be divided into three steps. 

1. Preprocess all of Wikipedia
2. Extracting article IDs for each category
3. Matching preprocessed Wikipedia articles 

#### Preprocessing Wikipedia
In order to preprocess all articles of Wikipedia you first need to download a 
Wikipedia dump from Download wiki dump from https://dumps.wikimedia.org/enwiki/latest/. 
In this project we used `enwiki-latest-pages-articles.xml.bz2`.

In `wikipedia_handlers.py` the function `generate_corpus()` needs to be called. 
You can either give in_file as argument corresponding to the path to Wikipedia dump
or place the exact Wikipedia dump we used in the `./data/` dir.

The function will preprocess all of Wikipedia articles into `./data/articles/`.

**Warning:** Preprocessing all of Wikipedia is very time consuming and will fill
up your local drive. 

#### Extracting category information
In `wikipedia_api_handler.py` the function `extract_page_ids_for_category()`
can be called. You can need to give the parameter `category` corresponding to 
the category you want to extract. The functions saves two files in `./data/categories/`, 
one file with article ids and one file with queued subcategories. 

It is possible to continue extracting article 
ids for a category using `continue_category_page_extraction()`. See the `__main__` method of `wikipedia_api_handler.py`
on how to call the functions.

**Sidenote:** You need to extract article ids seperately for each category in order
to abide with Wikipedias thumb rule of 1 request pr. second.

#### Final Dataset
If you have succesfully completed the two above steps, simply call the `create_dataset()` function
in `wikipedia_handlers.py`. This will generate the final dataset. 


### Running the models
To run each model seperate, simply call the `ml_models.py` with either argument
SVC or LSH specficying the specific model to be run. This will train the model
and evaluate the performance on test.

If you want to time both models, then the `performance_comparison.py` script may be called.

### Data Handling
The `DataHandler.py` file includes a python class `DataHandler`. The Datahandler 
controls all of the preprocessing and test/train splits depending on arguments. 


### Running the App
To run the app, first open the file `./wiki_catpred_app/predidiction/views.py` and specify
the `super_path` variable to match the root of the project on your local machine. 

Then simply go to directory `./wiki_catpred_app/` and execute 
`./manage.py runserver`. This will launch the app on localhost using port 8000. Go to the
URL `http://127.0.0.1:8000/prediction/` in order to predict a new document. 

Note that if you want to test an unseen document, you first need to extract the text
and tokenize the document using `tokenize_document()` function in `utils.py`. 
The `__main__` method in `utils.py` show how to tokenize an unseen document, so that it works with the webapp/models. 