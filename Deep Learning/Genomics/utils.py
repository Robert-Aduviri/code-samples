import os
import random
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

def filter_pvalue(df, pvalue_range):
    df = df[df['Pvalue'].apply(lambda x: x <= pvalue_range[0] or x >= pvalue_range[1])].copy()
    df['Target'] = df['Pvalue'].apply(lambda x: 1 if x < np.mean(pvalue_range) else 0)
    return df

def to_categorical(df, columns):
    for col in columns:
        df[col] = df[col].astype('category')
    return df

def load_fold(path, n_fold=0):
    folds = os.listdir(path)
    test_fold = folds[n_fold]
    print(f'Test fold: {test_fold}')
    test = pd.read_csv(f'{path}/{test_fold}', sep = ';')
    train = pd.DataFrame(columns = test.columns)
    for train_fold in [fold for fold in folds if fold != test_fold]:
        print(f'Train fold: {train_fold}')
        train = pd.concat((train, pd.read_csv(f'{path}/{train_fold}', sep = ';')))
    train = train.reset_index(drop=True)
    
    train = filter_pvalue(train, (0.01, 0.5))
    test = filter_pvalue(test, (0.01, 0.5))

    columns = ['CauseGene', 'EffectGene', 'Replicate', 'Treatment']
    train = to_categorical(train, columns)
    test = to_categorical(test, columns)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, test, test_fold

def gmean_score(y, y_pred):
    matrix = confusion_matrix(y, y_pred)
    tn = matrix[0][0]
    tp = matrix[1][1]
    fn = matrix[1][0]
    fp = matrix[0][1]
    return np.sqrt(tp / (tp + fn) * tn / (tn + fp))    

def true_positive_rate(y, y_pred):
    matrix = confusion_matrix(y, y_pred)
    tn = matrix[0][0]
    tp = matrix[1][1]
    fn = matrix[1][0]
    fp = matrix[0][1]
    return tp / (tp + fn)

def true_negative_rate(y, y_pred):
    matrix = confusion_matrix(y, y_pred)
    tn = matrix[0][0]
    tp = matrix[1][1]
    fn = matrix[1][0]
    fp = matrix[0][1]
    return tn / (tn + fp)

def evaluate_metrics(y, y_pred, y_pred_proba, model='', fold='', desc=''):
    df = pd.DataFrame()
    df.loc[0, 'Model'] = model
    df.loc[0, 'Fold'] = fold
    df.loc[0, 'ROC-AUC'] = roc_auc_score(y, y_pred_proba)
    df.loc[0, 'G-mean'] = gmean_score(y, y_pred)
    df.loc[0, 'F1-Score'] = f1_score(y, y_pred)
    df.loc[0, 'TPR'] = true_positive_rate(y, y_pred)
    df.loc[0, 'TNR'] = true_negative_rate(y, y_pred)
    df.loc[0, 'Accuracy'] = accuracy_score(y, y_pred)
    df.loc[0, 'Precision'] = precision_score(y, y_pred)
    df.loc[0, 'Recall'] = recall_score(y, y_pred)    
    df.loc[0, 'Logloss'] = log_loss(y, y_pred_proba)
    df.loc[0, 'Description'] = desc
    print('Confusion matrix:')
    print(confusion_matrix(y, y_pred))
    print(df)
    return df

def train_model(model, train, test):
    model.fit(train[[col for col in train.columns \
            if 'cause' in col or 'effect' in col]], train['Target'])
    y_pred = model.predict(test[[col for col in test.columns \
            if 'cause' in col or 'effect' in col]])
    y_pred_proba = model.predict_proba(test[[col for col in test.columns \
            if 'cause' in col or 'effect' in col]])[:,1]
    return model, y_pred, y_pred_proba

def train_model_sample(model, X_train, y_train, test):
    model.fit(X_train, y_train)
    y_pred = model.predict(test[[col for col in test.columns \
            if 'cause' in col or 'effect' in col]])
    y_pred_proba = model.predict_proba(test[[col for col in test.columns \
            if 'cause' in col or 'effect' in col]])[:,1]
    return model, y_pred, y_pred_proba

def evaluate_classifier_sample(path, model, results, bags=10, desc=''):
    for i in range(len(os.listdir(path))):
        train, test, test_fold = load_fold(path, i)
        print('Evaluating...')

        X_train, y_train = train[[col for col in train.columns \
            if 'cause' in col or 'effect' in col]], train['Target']
        y_pred = np.zeros(len(test))
        y_pred_proba = np.zeros(len(test))
        for j in range(bags):
            print('Bag No:', j)
            rus = RandomUnderSampler(random_state=random.randint(1, 1e9))
            X_resampled, y_resampled = rus.fit_sample(X_train, y_train)
            model, y_pred_, y_pred_proba_ = train_model_sample(model, 
                                            X_resampled, y_resampled, test)
            y_pred += y_pred_
            y_pred_proba += y_pred_proba_          
        
        y_pred = [int(round(x/bags)) for x in y_pred]
        y_pred_proba /= bags

        model_name = str(model.__class__).split('.')[-1].replace('>','').replace("'",'')
        result = evaluate_metrics(test['Target'], y_pred, y_pred_proba, 
                         model_name, test_fold.split('.')[0], desc)

        results = pd.concat((results, result)) 

    results = results.reset_index(drop=True)
    return results

def evaluate_classifier(path, model, results, desc=''):
    for i in range(len(os.listdir(path))):
        train, test, test_fold = load_fold(path, i)
        print('Evaluating...')

        model, y_pred, y_pred_proba = train_model(model, train, test)

        model_name = str(model.__class__).split('.')[-1].replace('>','').replace("'",'')
        result = evaluate_metrics(test['Target'], y_pred, y_pred_proba, 
                         model_name, test_fold.split('.')[0], desc)

        results = pd.concat((results, result)) 

    results = results.reset_index(drop=True)
    return results

################# DATA PREPROCESSING #####################

def outer_product(df):
    '''
    Creates new df_cause * df_effect columns
    Return dataframe
    '''
    df_cause = df[[col for col in df.columns if 'cause' in col]]
    df_effect = df[[col for col in df.columns if 'effect' in col]]
    df_columns = [f'outer-c{i+1:02}-e{j+1:02}' for i in range(df_cause.shape[1]) \
                                           for j in range(df_effect.shape[1])]
    df_data = np.array([np.outer(cause, effect) for cause, effect in \
                            zip(df_cause.values, df_effect.values)])
    df_data = np.reshape(df_data, (-1, df_cause.shape[1] * df_effect.shape[1]))
    df_data = pd.DataFrame(df_data, columns=df_columns)
    df = df[[col for col in df.columns \
                if 'cause' not in col and 'effect' not in col]]
    return pd.concat((df, df_data), 1)   
    
def group_data(df):
    '''
    Converts data format from (samples, attributes) 
    to (groups, samples_per_group, attributes)
    '''
    y = df['Target'].values
    y = np.reshape(y, (-1, 6))
    X = df[[col for col in df.columns if 'outer' in col]].values
    X = np.reshape(X, (-1, 6, X.shape[1]))
    return X, y   

def get_random_batch(X, y, sampler):
    '''
    Input / Output shapes
    X: (#elements, #treatments, #variables)
    y: (#elements, #treatments)
    '''
    X_reshaped = np.reshape(X, (-1, np.prod(X.shape[1:])))
    y_reshaped = np.array([np.prod(v) for v in y]) 
    X_resampled, y_resampled = sampler.fit_sample(X_reshaped, y_reshaped)
    X_resampled = np.reshape(X_resampled, (-1, X.shape[1], X.shape[2]))
    y_resampled = np.array([[v] * X.shape[1] for v in y_resampled])
    return X_resampled, y_resampled    

# TODO: use sklearn train_test_split
def train_val_split(X, y, val_size):
    '''
    Input / Output shapes
    X: (#elements, #treatments, #variables)
    y: (#elements, #treatments)
    '''
    merged = np.concatenate((X, y.reshape(-1, X.shape[1], 1)), 2)
    np.random.shuffle(merged)
    X, y = merged[:,:,:-1], merged[:,:,-1].reshape(-1, X.shape[1])
    return X[:-val_size], X[-val_size:], y[:-val_size], y[-val_size:]

def sklearn_train_test_split(X, y, test_size, random_state=None, shuffle=None, stratify=None):
    X_reshaped = np.reshape(X, (-1, np.prod(X.shape[1:])))
    y_reshaped = np.array([np.prod(v) for v in y]) 
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, 
            test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
    X_train = np.reshape(X_train, (-1, X.shape[1], X.shape[2]))
    X_test = np.reshape(X_test, (-1, X.shape[1], X.shape[2]))
    y_train = np.array([[v] * X.shape[1] for v in y_train])
    y_test = np.array([[v] * X.shape[1] for v in y_test])
    return X_train, X_test, y_train, y_test


##### CNN utils

from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Activation

def create_model():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(9, 9, 1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Conv2D(32, (3, 3), strides=2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Conv2D(32, (1, 1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
       
#     model.add(GlobalAveragePooling2D())

    model.add(Dense(1, activation='sigmoid'))
    
    adam = Adam()
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

##### Visualization ######

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set()

def plot_losses(train_loss, val_loss, train_acc, val_acc, scale):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ax1.plot(train_loss)
    ax1.plot([(x + 1) * scale - 1 for x in range(len(val_loss))], val_loss)
    ax1.legend(['train loss', 'validation loss'])
    ax2.plot(train_acc)
    ax2.plot([(x + 1) * scale - 1 for x in range(len(val_acc))], val_acc)
    ax2.legend(['train acc', 'validation acc'])

### Training utils

from keras.callbacks import ModelCheckpoint

def fit_base_learner(X_tra, y_tra, X_val, y_val, test_fold, X_test, y_test,
            keras_model, random_state=None, description=''):
    rus = RandomUnderSampler(random_state=random_state)
    X_train_batch, y_train_batch = get_random_batch(X_tra, y_tra, rus)
    X_val_batch, y_val_batch = get_random_batch(X_val, y_val, rus)  
    model = KerasClassifier(keras_model)
    
    checkpointer = ModelCheckpoint(filepath='_data/model.best.hdf5',
                               verbose=0, save_best_only=True)
    history = model.fit(X_train_batch.reshape(-1, 9, 9, 1), 
                        y_train_batch.reshape(-1), 
                         validation_data=(X_val_batch.reshape(-1, 9, 9, 1),
                                          y_val_batch.reshape(-1)),
                         batch_size=1024, epochs=100, verbose=0, 
                         callbacks=[checkpointer], shuffle=True)
    
    y_pred = model.predict(X_train_batch.reshape(-1, 9, 9, 1), batch_size=1024)
    y_pred_proba = model.predict_proba(X_train_batch.reshape(-1, 9, 9, 1), batch_size=1024)
    df_train = evaluate_metrics(y_train_batch.reshape(-1), y_pred, y_pred_proba[:,1],
                              'CNN', test_fold, description)
    
    y_pred = model.predict(X_val_batch.reshape(-1, 9, 9, 1), batch_size=1024)
    y_pred_proba = model.predict_proba(X_val_batch.reshape(-1, 9, 9, 1), batch_size=1024)
    df_val = evaluate_metrics(y_val_batch.reshape(-1), y_pred, y_pred_proba[:,1],
                              'CNN', test_fold, description)
    
    y_pred = model.predict(X_test.reshape(-1, 9, 9, 1), batch_size=1024)
    y_pred_proba = model.predict_proba(X_test.reshape(-1, 9, 9, 1), batch_size=1024)
    df_test = evaluate_metrics(y_test.reshape(-1), y_pred, y_pred_proba[:,1],
                              'CNN', test_fold, description)
    return df_train, df_val, df_test, history, model

def append_results(results, df_train, df_val, df_test):
    col_order = ['Results'] + [col for col in df_train.columns if col != 'Results']
    df_train['Results'] = 'train'
    df_val['Results'] = 'valid'
    df_test['Results'] = 'test'
    return pd.concat((results, df_train, df_val, df_test))[col_order]