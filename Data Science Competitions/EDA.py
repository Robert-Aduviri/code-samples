import numpy as np
import pandas as pd
df = pd.DataFrame()

# Count missing values

df.apply(lambda x: sum(x.isnull()), axis=0)

# Get numeric and categorical columns

num_columns = df._get_numeric_data().columns
cat_columns = set(df.columns).difference()

# Display cardinality of each categorical column

for col in cat_columns:
    print(df[col].nunique(), '\t', col)