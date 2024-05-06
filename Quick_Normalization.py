import pandas as pd
import numpy as np

data = {
    'Ctrl': [61.0041999816895 ,64.0334014892578 ,87.7265014648438 ,58.2648010253906 ,59.9216003417969 ,72.1035995483399 ,66.1695022583008 ,61.0844993591309 ,87.4268035888672 ,57.7593994140625],
    '15min': [153.669998168945 ,125.580001831055 ,136.888000488281 ,106.981002807617 ,90.092903137207]
}
# Find maximum length
max_length = max(len(v) for v in data.values())

# Fill NaN for missing values in each list
for key in data.keys():
    data[key] += [np.nan] * (max_length - len(data[key]))

df = pd.DataFrame(data)

# Divide each value in '15min' column by its corresponding value in 'Ctrl' column
ctrl_mean = df['Ctrl'].mean()

# Normalize the control column so that its mean is 1
df['Ctrl'] = df['Ctrl'] / ctrl_mean

# Divide each value in the other column by its corresponding control mean
df['15min'] = df['15min'] / ctrl_mean

print(df)
