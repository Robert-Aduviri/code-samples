import re
import numpy as np
import pandas as pd
df = pd.DataFrame({
    'date': pd.Series([], dtype='str'),
    'number': pd.Series([], dtype='float'),
    'cat1': pd.Series([], dtype='str'),
    'cat2': pd.Series([], dtype='str')
})
    
# Date parsing %Y-%m-%d %H:%M:%S.%f

from datetime import datetime

def parse_dates(df, column, string_format='%d-%m-%Y'):
    df[f'{column}-day'] = df[column].apply(
        lambda x: datetime.strptime(x, string_format).day)
    df[f'{column}-month'] = df[column].apply(
        lambda x: datetime.strptime(x, string_format).month)
    df[f'{column}-year'] = df[column].apply(
        lambda x: datetime.strptime(x, string_format).year)
    df[f'{column}-weekday'] = df[column].apply(
        lambda x: datetime.strptime(x, string_format).weekday())
    return df.drop(column, axis=1)

df = parse_dates(df, 'date')

# Percentile clipping (Winsorizing)

q_min = df.number.quantile(0.01)
q_max = df.number.quantile(0.99)
df['number'] = df.number.apply(lambda x: np.clip(x, q_min, q_max))

# Label encoding

from sklearn.preprocessing import LabelEncoder
cat_columns = ['cat1', 'cat2']
label_encoders = {col: LabelEncoder().fit(df[col]) \
                        for col in cat_columns}
for col in cat_columns:
    df[col] = label_encoders[col].transform(df[col])

# MinMax scaling

from sklearn.preprocessing import MinMaxScaler
numeric_columns = ['number']
scalers = {col: MinMaxScaler().fit(df[col]) \
                        for col in numeric_columns}
for col in numeric_columns:
    df[col] = scalers[col].transform(df[col])
    
# Add date features
# From https://github.com/fastai/fastai/blob/master/fastai/structured.py

def add_datepart(df, fldname, drop=True, time=False):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.
    Examples:
    ---------
    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df
        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13
    >>> add_datepart(df, 'A')
    >>> df
        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed
    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800
    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200
    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600
    """
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)
        
add_datepart(df, 'date')