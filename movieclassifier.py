import pandas as pd
import requests
import json

key = open('../api-key/omdb.txt', 'r').read()
searchlist = ['love']\

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
        response_title = requests.get('http://www.omdbapi.com/?apikey={}&s={}'.format(key, word))
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
    moviedf.to_excel('test.xlsx', index=False)
    return moviedf

load_data(key, searchlist)