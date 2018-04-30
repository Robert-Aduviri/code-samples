from flask import Flask
app = Flask(__name__)

import sys
sys.path.insert(0, '../')
from utils import preprocess, read_model, read_data, create_inverted_index, search

df_profiles = None
word2vec_profiles = None
inverted_index_profiles = None

@app.route('/ping')
def ping():
    return 'pong!'

@app.route('/preprocess')
def preprocess_string():
    words = 'Programador Python avanzando reconocimiento de KPIs y certificación ORACLE'
    return preprocess(words)

@app.route('/query')
def query():
    query = 'Programador Python avanzando reconocimiento de KPIs y certificación ORACLE'
    return str(search(query, word2vec_profiles, inverted_index_profiles))

import os

if __name__ == '__main__':
    root_dir = '../../_data/s3/'
    df_profiles = read_data(root_dir + 'df_profiles_25k.csv')
    word2vec_profiles = read_model(root_dir + 'word2vec_profiles_25k') 
    inverted_index_profiles = create_inverted_index(df_profiles, word2vec_profiles)
    app.run()