import re
import subprocess as sp
import multiprocessing as mp

from pathlib import Path
from tqdm import tqdm_notebook # or tqdm
import pandas as pd
import numpy as np

# original = Path('/media/ks/GianR/Original/sopranos')
# new = Path('/media/ks/KS/new-sopranos/SOpranos-catalogo')

###### Preprocessing #######

def list_songs(root_path):
    """
    DFS over root_path tree  
    Args:
        root_path: pathlib.Path
    Returns:
        songs: [pathlib.Path]
    """
    songs = []
    frontier = list(root_path.iterdir())
    while len(frontier) > 0:
        u = frontier.pop()
        for v in u.iterdir():
            if v.is_dir():
                frontier.append(v)
            elif v.is_file():
                songs.append(v)
    return songs

def get_dataframe(song_paths):
    """
    Reformat list as pandas.DataFrame
    Args: 
        song_paths: [pathlib.Path]
    Returns:
        df: pandas.DataFrame
    """
    df = pd.DataFrame()
    df['path'] = song_paths
    df['name'] = df.path.apply(lambda x: x.name)
    df['extension'] = df.path.apply(lambda x: x.suffix)
    return df

def preprocess_dataframe(df):
    """
    Clean pandas.DataFrame fields
    Args:
        df: pandas.DataFrame
    Returns:
        df: pandas.DataFrame
    """
    media_extensions = ['.avi', '.cdg', '.dat', '.mkv', '.mp3', '.mp4', '.mpg', '.wav', '.wma', '.wmv']
    df['extension'] = df.extension.apply(str.lower)
    # remove extension
    df['name'] = df.apply(lambda row: \
                    row['name'].lower()[:-len(row['extension'])], axis=1)
    # clean 'cancion.mp4.mp3' => 'cancion.mp3'
    df['name'] = df.name.apply(lambda x: x[:-4] if x[-4:] in media_extensions else x) 
    df = df[df.extension.apply(lambda x: x in media_extensions)].copy()
    
    pattern = re.compile('([eij]-?\d+-?\d*-?[a-z]?)\Z') # find 'e-2600-14-r'
    def get_code(song_name):
        matches = re.findall(pattern, song_name.replace(' ',''))
        return matches[0] if len(matches) > 0 else np.nan
    df['code'] = df.name.apply(get_code)
    
    return df.reset_index(drop=True)
    
def copy_files(file_paths, target_path):
    """
    Copy files in file_paths to target_path
    Check progress with `pidof cp | wc -w`
    Args:
        file_pahts: [pathlib.Path]
        target_path: pathlib.Path
    """
    def copy(index):
        sp.Popen(['cp', file_paths[index], target_path])        
    pool = mp.Pool()
    for _ in tqdm_notebook(pool.imap_unordered(copy, range(len(file_paths))), 
                            total=len(file_paths)):
        pass
    pool.close()
    pool.join()
    
def delete_files(file_paths):
    """
    Delete files in file_paths
    Check progress with `pidof rm | wc -w`
    Args:
        file_pahts: [pathlib.Path]
    """
    def delete(index):
        sp.Popen(['rm', '-f', file_paths[index]])        
    pool = mp.Pool()
    for _ in tqdm_notebook(pool.imap_unordered(delete, range(len(file_paths))), 
                            total=len(file_paths)):
        pass
    pool.close()
    pool.join()
    
def get_comparison_chart(df1, df2):
    """
    Get comparison chart
    Args:
        df1: pandas.DataFrame
        df2: pandas.DataFrame
    Returns:
        comparison_chart: pandas.DataFrame
    """
    comparison_chart = pd.concat((df1.extension.value_counts(), 
                        df2.extension.value_counts()), axis=1).fillna(0).astype(int)
    comparison_chart.columns = ['original', 'new']
    return comparison_chart

###### Extract audio features #######


# original_df = preprocess_dataframe(get_dataframe(list_songs(original)))
# new_df = preprocess_dataframe(get_dataframe(list_songs(new)))

# original_df = original_df.drop_duplicates('name')
# new_df = new_df.drop_duplicates('name')

# copy_files(list(original_df[original_df.code.isnull()].path),
#            '/home/ks/Desktop/song files diff/original_nocode')
# remove_files(list(original_df[original_df.code.isnull()].path))


