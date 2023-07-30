import pandas as pd
import requests
import json
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

key = open('../api-key/omdb.txt', 'r').read()
searchlist = ['love']

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

            # add data to the final dataframe
            moviedf = pd.concat([moviedf, parse_data])
    return moviedf


def process_num(df, num_cols):
    df_num = df[num_cols]
    # impute numerical values with the median
    imputer = SimpleImputer(strategy="median")
    df_num = imputer.fit_transform(df_num)

    # apply feature scaling 
    df_num = StandardScaler().fit_transform(df_num)
    return df_num

def process_cat(df, cat_cols):
    df_cat = df[cat_cols]
    # Drop columns with more than 10% missing values
    missing_counts = df_cat.isnull().sum() * 100 / len(df)
    d = {k:v for (k,v) in missing_counts.items() if v>10}
    df_cat.drop(d.keys(), axis=1, inplace=True)

    # convert string data to numbers
    cat_encoder = OneHotEncoder()
    df_cat_encoded = cat_encoder.fit_transform(df_cat)
    return df_cat_encoded


def preprocess_df(df):
    numeric_cols = ['Runtime', 'Metascore', 'imdbRating', 'imdbVotes', 'BoxOffice']
    cat_cols = ['Genre', 'Language', 'Country']
    text_cols = ['Title', 'Plot']
    date_cols = ['Year']

    num_pipeline = Pipeline([
    ('drop_columns', Drop_miss_cols()),
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
    ('drop_columns', Drop_miss_cols()),
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
    ])



    # allcols = 
    # make sure numeric columns are numeric
    # moviedf['Year'] = moviedf['Year'].astype(int)
    # moviedf['Runtime'] = moviedf.Runtime.str.extract('(\d+)')
    # moviedf['Metascore'] = moviedf.Metascore.str.extract('(\d+)')
    # moviedf['BoxOffice'] = moviedf.Metascore.str.extract('(\d+)')
    print(df.info())
    # moviedf.to_excel('test.xlsx', index=False)
    return moviedf

# moviedf = load_data(key, searchlist)
moviedf = pd.read_excel('test.xlsx')
processed_df = preprocess_df(moviedf)