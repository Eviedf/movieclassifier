import pandas as pd
import requests
import json
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


key = open('../api-key/omdb.txt', 'r').read()
searchlist = ['love', 'ball', 'war']

# sklearn transformer to select columns
class SelectColumnsTransformer():
    def __init__(self, columns=None):
        self.columns = columns
    
    
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        cpy_df = X[self.columns].copy()
        return cpy_df

    
# sklearn transformer to remove columns with more than 10% missing values
class Drop_miss_cols(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
    # Drop columns with more than 10% missing values
        missing_counts = X.isnull().sum() * 100 / len(X)
        d = {k:v for (k,v) in missing_counts.items() if v>10}
        X.drop(d.keys(), axis=1, inplace=True)
        return X
    
# sklearn transformer to extract numbers from numeric columns
class Extract_numbers(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for col in list(X.columns):
            X.loc[:,col] = X[col].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int)
        return X

# helper functions
def getratings(ratinglist):
    """ 
     TODO: comment this helper function
       """
    returnlist = []
    for rating in ratinglist:
        dict_r = dict(rating)
        return_r = [dict_r['Source'], dict_r['Value']]
        returnlist.append(return_r)
    return returnlist

def process_text(text):
    text = str(text).lower()
    text = re.sub(
        f"[{re.escape(string.punctuation)}]", " ", text
    )
    text = " ".join(text.split())
    return text

def get_col_data(x, cols):
    return x[cols]

def get_text_data(x, col):
    return x[col].values.ravel()


def load_data(key, searchlist):
    """ 
    load_data extracts movie data from OMDB API, based on a list of words provided in searchlist
    :param key: apikey to access the api
    :param searchlist: list of stings to search for in movie titles
    :return:  for each word in searchlist, 10 movies with the word in its title and corresponding omdb data
    """
    # initialize empty dataframe
    moviedf = pd.DataFrame()
    # retrieve the data from api and get 10 results based on keyword 
    for word in searchlist: 
        response_title = requests.get('http://www.omdbapi.com/?apikey={}&s={}&type=movie'.format(key, word))
        title_txt = response_title.text
        parse_title = json.loads(title_txt)
        titledf = parse_title['Search']
        titledf = pd.DataFrame.from_dict(titledf)
        titles = titledf['Title']
        # for each movietitle, acquire extra data
        for title in titles:
            response_data = requests.get('http://www.omdbapi.com/?apikey={}&t={}&plot=short'.format(key, title))
            data_txt = response_data.text
            parse_data = pd.read_json(data_txt)

            # convert multiple rows per movie into 1 row per movie by groupby and applying a function to 'Ratings' column
            groupby_cols = list(parse_data.columns)
            groupby_cols.remove('Ratings')
            parse_data = (parse_data.groupby(groupby_cols)
            .agg({'Ratings': lambda x: getratings(x)},axis=1)
            .reset_index())

            parse_data['search_word'] = word
            # add data to the final dataframe
            moviedf = pd.concat([moviedf, parse_data])
    # make sure there are no duplicates
    moviedf.reset_index(drop=True, inplace=True)
    moviedf = moviedf.loc[moviedf.astype(str).drop_duplicates().index]
    moviedf.to_excel('test.xlsx')
    return moviedf

def model_df(df):
    df['clean_text'] = df.Plot.map(process_text)

    num_cols = ['Runtime', 'Metascore', 'imdbRating', 'imdbVotes', 'BoxOffice']
    text_cols = ['clean_text', 'Title']
    X_cols = text_cols + num_cols
    X = df[X_cols]
    # transform target variable
    multilabel_binarizer = MultiLabelBinarizer()
    genre = [x.split(' ') for x in list(df['Genre'])]
    y = multilabel_binarizer.fit_transform(genre)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    transformer_text = FunctionTransformer(get_text_data, kw_args={'col' : 'clean_text'})
    transformer_num = FunctionTransformer(get_col_data, kw_args={'cols' : num_cols})

    num_pipeline = Pipeline([
    # ('selector', SelectColumnsTransformer(num_cols)),
    # ('drop_columns', Drop_miss_cols()),
    ('selector', transformer_num),
    ('extract_numbers', Extract_numbers()),
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
    ])

    text_pipeline = Pipeline(
    [   
        ('selector', transformer_text),
        ("vect", TfidfVectorizer(stop_words='english', analyzer='word')),
    ]
)

    data_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('numeric_feat', num_pipeline),
            ('text_feat', text_pipeline)
        ])),
        ('clf', OneVsRestClassifier(LinearSVC(dual=True)))
    ])

    print(y_train)
    # fit model on train data
    data_pipeline.fit(X_train, y_train)
    # make predictions for validation set
    y_pred = data_pipeline.predict(X_test)

    print(f1_score(y_test, y_pred, average="micro"))

    return X_test, y_test, y_pred

def plot_PCA(X, y):
    transformer_text = FunctionTransformer(get_text_data, kw_args={'col' : 'clean_text'})
    transformer_num = FunctionTransformer(get_col_data, kw_args={'cols' : ['Runtime', 'Metascore', 'imdbRating', 'imdbVotes', 'BoxOffice']})

    num_pipeline = Pipeline([
    # ('selector', SelectColumnsTransformer(num_cols)),
    # ('drop_columns', Drop_miss_cols()),
    ('selector', transformer_num),
    ('extract_numbers', Extract_numbers()),
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
    ])

    text_pipeline = Pipeline(
    [   
        ('selector', transformer_text),
        ("vect", TfidfVectorizer(stop_words='english', analyzer='word')),
    ]
)

    preprocess_pipeline = Pipeline([
        ('numeric_feat', num_pipeline),
        ('text_feat', text_pipeline),
        ('pca', PCA(n_components=2))
    ])

    X_pca = preprocess_pipeline.fit_transform(X)
    print(X_pca)





# moviedf = load_data(key, searchlist)
moviedf = pd.read_excel('test.xlsx')
# print(moviedf)
X_test, y_test, y_pred= model_df(moviedf)
plot_PCA(X_test, y_test)



