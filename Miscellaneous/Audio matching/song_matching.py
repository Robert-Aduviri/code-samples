import os
import json
import subprocess as sp
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def extract_song_features(file_path):
    f = os.path.splitext(file_path)[0]
    sp.call(['essentia_streaming_extractor_music', 
             os.path.join(f + '.wav'),
             os.path.join(f + '.json')])
    
def extract_songs_features(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    for f in files:
        print(os.path.join(folder_path, f))
        extract_song_features(os.path.join(folder_path, f))
        
def load_song_features(file_path):
    data = {}
    features = json.load(open(file_path))    

    # Low level
    data['average_loudness'] = features['lowlevel']['average_loudness']
    for feat in ['spectral_centroid', 'spectral_flux', 
                 'spectral_entropy', 'spectral_energy', 'pitch_salience']:
        data[feat] = features['lowlevel'][feat]['mean']
    for i, mfcc in enumerate(features['lowlevel']['mfcc']['mean']):
        data[f'mfcc-{i:02}'] = mfcc
    for i, spectral_contrast in enumerate(features['lowlevel']['spectral_contrast_coeffs']['mean']):
        data[f'spectral_contrast_coeff-{i:02}'] = spectral_contrast

    # Metadata
    data['length'] = features['metadata']['audio_properties']['length']

    # Rhythm
    data['beats_count'] = features['rhythm']['beats_count']
    data['bpm'] = features['rhythm']['bpm']
    data['danceability'] = features['rhythm']['danceability']

    # Tonal
    data['chords_key'] = features['tonal']['chords_key']
    data['chords_scale'] = features['tonal']['chords_scale']
    data['chords'] = data['chords_key'] + data['chords_scale']
    return data

def load_songs_features(folder_path):
    files = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('.wav')]
    data = {}
    for f in files:
        data[f] = load_song_features(os.path.join(folder_path, f + '.json'))
    return pd.DataFrame.from_dict(data, orient='index')

def get_comparison_df(df):
    '''
    df: song features dataframe
    returns song pairs feature comparison 
    '''
    comparison_data = {}
    for song1 in df.index:
        for song2 in df.index:
            comparison_data[(song1, song2)] = {}
            for feat in df.columns:
                if feat in ['chords_key', 'chords_scale', 'chords']:
                    comparison_data[(song1, song2)][feat] = int(df.loc[song1][feat] == df.loc[song2][feat])
                else:
                    comparison_data[(song1, song2)][feat] = abs(df.loc[song1][feat] - df.loc[song2][feat]) ** 2
            if song1[-6:] == song2[-6:]:
                print(song1, song2)
                comparison_data[(song1, song2)]['same_song'] = 1
            else:
                comparison_data[(song1, song2)]['same_song'] = 0
    return pd.DataFrame.from_dict(comparison_data, orient='index')

def get_feature_weights(df, l1_reg=1):
    '''
    df: song pairs feature comparison dataframe
    returns feature weights dictionary
    '''
    lr = LogisticRegression(penalty='l1', C=l1_reg)
    lr.fit(df.drop('same_song', axis=1), df['same_song'])
    return {feature: weight for feature, weight in \
                 zip(df.drop('same_song', axis=1).columns, lr.coef_[0])}

def get_similarity(song1, song2, param_weights):
    comparison_log = []
    s = 0
    for feat in song1.keys():
        if feat in ['chords_key', 'chords_scale', 'chords']:
            comparison_log.append([feat, param_weights[feat], 
                                   int(song1[feat] == song2[feat]),
                                   param_weights[feat] * int(song1[feat] == song2[feat])])
            s += param_weights[feat] * int(song1[feat] == song2[feat])
        else:
            comparison_log.append([feat, param_weights[feat], 
                                   abs(song1[feat] - song2[feat]),
                                   param_weights[feat] * abs(song1[feat] - song2[feat])])
            s += param_weights[feat] * abs(song1[feat] - song2[feat])
    return 1 / (1 + np.exp(-s)), comparison_log

def get_similarity_from_files(song1_path, song2_path, param_weights):
    song1 = load_song_features(song1_path)
    song2 = load_song_features(song2_path)
    return get_similarity(song1, song2, param_weights)

def most_similar(song_path, param_weights, song_db=None, song_folder=None):
    song_query = load_song_features(song_path)
    results = []
    if song_db is not None:
        for song_name, song in song_db.iterrows():
            results.append([song_name, get_similarity(song_query, song.to_dict(), param_weights)[0]])
    elif song_folder:
        files = [os.path.splitext(f)[0] for f in os.listdir(song_folder) if f.endswith('.json')]
        for song_name in files:
            song = load_song_features(os.path.join(song_folder, song_name + '.json'))
            results.append([song_name, get_similarity(song_query, song, param_weights)[0]])
    else:
        return None
    return sorted(results, key=lambda x: x[1], reverse=True)
            
        
            