import json
import re
from collections import Counter, defaultdict

import pandas as pd
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from tqdm import tqdm_notebook, tqdm

from oov_utils import translate, spell_checking

####### LOADING ######

def lightweight_json_reader(file_name, column_names, print_every=1000000):
    '''
    Valid for json and json lines files, as long each data point is in a single row
    Input: a list of attributes to extract from the file
    Output: a list of data points, where each data point is a list of attributes in the same order as column_names
    '''
    data = []
    with open(file_name) as f:
        for idx, line in enumerate(f):
            if len(line) > 5: # To skip brackets in json files
                row = json.loads(line.strip().strip(',')) # To skip commas in json files
                data.append([row[column] if column in row else '' for column in column_names])
            if idx % print_every == 0: # Log progress
                print(idx, end=' ')
    print()
    return data

####### CLEANING ######

def drop_bad_descriptions(df, description_column, hscode_column):
    description_count = df.groupby(description_column)[hscode_column].nunique()
    bad_descriptions = set(description_count[description_count > 1].index)
    return df[df[description_column].apply(lambda x: x not in bad_descriptions)].copy()

####### OOV ######

def get_oov(descriptions, vocab):
    '''
    Input:
        description: array-like list of descriptions
        vocab: set-like vocabulary (such that supports `token in vocab` operation)
    Output:
        OOV: Counter containing OOV words
    '''
    OOV = Counter()
    pattern = re.compile('[^a-zA-Z]+', re.UNICODE)
    for description in tqdm_notebook(descriptions):
        description = pattern.sub(' ',description).strip()
        # remove words with less than 3 characters
        description = ' '.join([w for w in description.split() if len(w)>=2]) 
        if(description.isspace() or len(description) == 0):
            continue
        filtered_tokens = [token for token in description.split() if not token in vocab]
        OOV.update(filtered_tokens)
    return OOV

####### LEMMATIZATION ######

def lemmatize(descriptions):
    tqdm_notebook().pandas()
    nlp = spacy.load('en')
    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
    return descriptions.progress_apply(lambda desc: \
                ' '.join([lemmatizer(token.text, token.pos_)[0] \
                for token in nlp(desc)]))

####### MAIN PIPELINE ######

section_labels = {
    1: 'LIVE ANIMALS; ANIMAL PRODUCTS',
    2: 'VEGETABLE PRODUCTS',
    3: 'ANIMAL OR VEGETABLE FATS AND OILS AND THEIR CLEAVAGE PRODUCTS; PREPARED EDIBLE FATS; ANIMAL OR VEGETABLE WAXES',
    4: 'PREPARED FOODSTUFFS; BEVERAGES, SPIRITS AND VINEGAR; TOBACCO AND MANUFACTURED TOBACCO SUBSTITUTES',
    5: 'MINERAL PRODUCTS',
    6: 'PRODUCTS OF THE CHEMICAL OR ALLIED INDUSTRIES',
    7: 'PLASTICS AND ARTICLES THEREOF; RUBBER AND ARTICLES THEREOF',
    8: 'RAW HIDES AND SKINS, LEATHER, FURSKINS AND ARTICLES THEREOF; SADDLERY AND HARNESS; TRAVEL GOODS, HANDBAGS AND SIMILAR CONTAINERS; ARTICLES OF ANIMAL GUT (OTHER THAN SILK-WORM GUT)',
    9: 'WOOD AND ARTICLES OF WOOD; WOOD CHARCOAL; CORK AND ARTICLES OF CORK; MANUFACTURES OF STRAW, OF ESPARTO OR OF OTHER PLAITING MATERIALS; BASKETWARE AND WICKERWORK',
    10: 'PULP OF WOOD OR OF OTHER FIBROUS CELLULOSIC MATERIAL; RECOVERED (WASTE AND SCRAP) PAPER OR PAPERBOARD; PAPER AND PAPERBOARD AND ARTICLES THEREOF',
    11: 'TEXTILES AND TEXTILE ARTICLES',
    12: 'FOOTWEAR, HEADGEAR, UMBRELLAS, SUN UMBRELLAS, WALKING-STICKS, SEAT-STICKS, WHIPS, RIDING-CROPS AND PARTS THEREOF; PREPARED FEATHERS AND ARTICLES MADE THEREWITH; ARTIFICIAL FLOWERS; ARTICLES OF HUMAN HAIR',
    13: 'ARTICLES OF STONE, PLASTER, CEMENT, ASBESTOS, MICA OR SIMILAR MATERIALS; CERAMIC PRODUCTS; GLASS AND GLASSWARE',
    14: 'NATURAL OR CULTURED PEARLS, PRECIOUS OR SEMI-PRECIOUS STONES, PRECIOUS METALS, METALS CLAD WITH PRECIOUS METAL AND ARTICLES THEREOF; IMITATION JEWELLERY; COIN',
    15: 'BASE METALS AND ARTICLES OF BASE METAL',
    16: 'MACHINERY AND MECHANICAL APPLIANCES; ELECTRICAL EQUIPMENT; PARTS THEREOF; SOUND RECORDERS AND REPRODUCERS, TELEVISION IMAGE AND SOUND RECORDERS AND REPRODUCERS, AND PARTS AND ACCESSORIES OF SUCH ARTICLES',
    17: 'VEHICLES, AIRCRAFT, VESSELS AND ASSOCIATED TRANSPORT EQUIPMENT',
    18: 'OPTICAL, PHOTOGRAPHIC, CINEMATOGRAPHIC, MEASURING, CHECKING, PRECISION, MEDICAL OR SURGICAL INSTRUMENTS AND APPARATUS; CLOCKS AND WATCHES; MUSICAL INSTRUMENTS; PARTS AND ACCESSORIES THEREOF',
    19: 'ARMS AND AMMUNITION; PARTS AND ACCESSORIES THEREOF',
    20: 'MISCELLANEOUS MANUFACTURED ARTICLES',
    21: "WORKS OF ART, COLLECTORS' PIECES AND ANTIQUES"
}

sections = [1,6,15,16,25,28,39,41,44,47,50,64,68,71,72,84,86,90,93,94,97,100]
section_map = {chapter: idx+1 for idx in range(len(sections)-1) \
               for chapter in range(sections[idx], sections[idx+1])}

def clean(df, column_names):
    id_col, desc_col, hscode_col, status_col = column_names
    print('Cleaning data...')
    print('   Dropping null records...')
    df = df.dropna(subset=[desc_col, hscode_col])
    print('   Dropping ambiguous descriptions...')
    df = drop_bad_descriptions(df, desc_col, hscode_col)    
    print('   Dropping duplicates...')
    df = df.drop_duplicates(subset=[desc_col, hscode_col])
    
    print('Adding targets...')
    df = df[df[hscode_col].apply(lambda x: len(x)>=6 and x[:6].isdigit())]
    df['chapter'] = df[hscode_col].apply(lambda x: x[:2])
    df['heading'] = df[hscode_col].apply(lambda x: x[:4])
    df[hscode_col] = df[hscode_col].apply(lambda x: x[:6])
    df = df[df['chapter'].apply(lambda x: int(x) in section_map)]
    df['section'] = df['chapter'].apply(lambda x: section_map[int(x)])
    
    return df

def split_camel_case(oov, oovToWords, log_every=100):
    remainingOov = []
    pattern = re.compile('(?!^)([A-Z][a-z]+)', re.UNICODE)
    for word in tqdm_notebook(oov):
        if(len(oovToWords[word]) != 0):
            continue
        splitted = pattern.sub(r' \1', word).split()
        if(len(splitted) == 0 or len(splitted) == 1):
            remainingOov.append(word)
            continue
        for w in splitted:
            oovToWords[word].append(w)
    return remainingOov

def handleOOV(df, column_names, vocab):
    id_col, desc_col, hscode_col, status_col = column_names
    print('Handling OOV words...')
    print('   Getting OOV words...')
    OOV = get_oov(df[desc_col], vocab)
    oovToWords = defaultdict(list) 
    print('   Splitting camelcase words...')
    OOV = split_camel_case(OOV, oovToWords, log_every=100000)
    # to lower before the next ones
    # OOV = translate(OOV, oovToWords, log_every=100000)
    # OOV = spell_checking(OOV, oovToWords, log_every=100000)
    print('   Recovering OOV words...')
    df['parsed_description'] = df[desc_col].apply(lambda desc: \
                                    ' '.join(' '.join(oovToWords[word]) \
                                    if len(oovToWords[word])>0 else word \
                                    for word in desc.split()))
    return df, OOV

def preprocessing(df, column_names, vocab):
    tqdm_notebook().pandas()
    id_col, desc_col, hscode_col, status_col = column_names
    
    print('Preprocessing description...')
    pattern = re.compile('[^a-z]+', re.UNICODE)
    
    print('   Removing non-alphabetic characters...')
    df['parsed_description'] = df.parsed_description.progress_apply(lambda x: \
                                pattern.sub(' ', x.lower()).strip())
    # print('   Lemmatizing...')
    # df['parsed_description'] = lemmatize(df.parsed_description)
    
    print('   Removing words shorter than 2 characters...')
    df['parsed_description'] = df.parsed_description.progress_apply(lambda x: \
                            ' '.join([w for w in x.split() if len(w)>=2])) 
    OOV = None
    # OOV = get_oov(df.parsed_description, vocab)
    
    print('   Removing remaining OOV...')
    df['parsed_description'] = df.parsed_description.progress_apply(lambda x: \
                            ' '.join([w for w in x.split() if w in vocab]))
    
    print('   Removing sentences with less than 3 words...')
    df = df[df.parsed_description.apply(lambda x: len(x.split())>=3)]
    
    return df, OOV

def pipeline(file_name, column_names, vocab):
    '''
    column_names format is [id, description, hscode, status]
    '''
    id_col, desc_col, hscode_col, status_col = column_names
    data = lightweight_json_reader(file_name, column_names)
    df = pd.DataFrame(data, columns=columns)
    
    df = clean(df, column_names)    
    df, OOV = handleOOV(df, column_names, vocab)    
    df = preprocessing(df, column_names, vocab)
       
    return df

###### TRAIN ######

