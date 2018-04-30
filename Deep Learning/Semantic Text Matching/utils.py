import re
from nltk.corpus import stopwords
from unidecode import unidecode
from collections import Counter
import difflib, jellyfish
from fuzzywuzzy import fuzz
import pandas as pd
import gensim

sp_stopwords = set(stopwords.words('spanish'))

def preprocess(x):    
    x = re.sub('[\n\r\t/]', ' ', x)
    x = re.sub('[{0}]'.format(re.escape("%0123456789&+@#,_–°´¡!-()=<>.:;?|'“”’*·•\"")), ' ', x)
    x = re.sub('[{0}]'.format(re.escape("$%()+0123456789;=?")), ' ', x)
    x = x.lower()
    return ' '.join([unidecode(w) for w in x.split() \
                     if len(w) > 2 and w not in sp_stopwords])

#### String matching utils

def difflib_match(word1, word2):
    return difflib.SequenceMatcher(None, word1, word2).ratio()

def levenshtein_match(word1, word2):
    return 1.0 - jellyfish.levenshtein_distance(word1, word2)

def fuzzy_match(word1, word2):
    return fuzz.ratio(word1, word2) / 100

def predict_keyword(keyword, vocabulary, match_function):
    s = vocabulary.apply(lambda x: match_function(x, keyword))
    idx_max = s.idxmax()
    match = s.loc[idx_max]
    predicted = vocabulary.loc[idx_max]
    return predicted, match

def best_string_match(keyword, vocabulary):
    vocabulary = pd.Series(list(vocabulary))
    pred1, match1 = predict_keyword(keyword, vocabulary, difflib_match)
    pred2, match2 = predict_keyword(keyword, vocabulary, fuzzy_match)
    pred3, match3 = predict_keyword(keyword, vocabulary, levenshtein_match)
    # majority wins
    if pred1 == pred2: return pred1, match1
    if pred2 == pred3: return pred2, match2
    if pred1 == pred3: return pred1, match3
    # else most confident one wins
    # TODO: return null if there isn't any good match (given threshold)
    return [pred1, pred2, pred3][np.argmax([match1, match2, match3])]

#### Inverted index utils

def create_inverted_index(df, model):
    '''
    df: the database
    model: a corresponding Word2Vec model
    '''
    print('Creating inverted index...')
    inverted_index = {}
    for index, description in zip(df.index, df['preprocessed']):
        keyword_counts = Counter([word for word in description.split() \
                                     if word in model.wv.vocab.keys()])
        for word in keyword_counts:
            if word not in inverted_index:
                inverted_index[word] = Counter()
            # TODO: store relative frequency
            inverted_index[word][index] = keyword_counts[word]
    return inverted_index

def search(keywords, model, idx2doc, min_similarity=0.80, 
           max_similar_words=10, max_documents_per_keyword=20,
           max_results=10):
    '''
    keywords: a keywords string
    model: a Word2Vec model
    idx2doc: a corresponding inverted_index
    '''
    keywords = preprocess(keywords)
    results = Counter()
    for keyword in keywords.split():
        if keyword not in model.wv.vocab.keys():
            # TODO: consider null results if there isn't any good match
            keyword = best_string_match(keyword, model.wv.vocab.keys())[0]
        # TODO: consider keyword TF-IDF for ranking
        for word, similarity in [(keyword, 1)] + model.wv.most_similar(keyword, 
                                                  topn=max_similar_words):
            if similarity > min_similarity:
                docs = idx2doc[word].most_common(max_documents_per_keyword)
                # TODO: consider TF in doc and similarity for ranking
                results.update([doc[0] for doc in docs])
    return results.most_common(max_results)

def read_data(path):
    print(f'Reading data in {path}...')
    return pd.read_csv(path, encoding='latin-1')

def read_model(path):
    print(f'Reading model in {path}...')
    return gensim.models.Word2Vec.load(path)
