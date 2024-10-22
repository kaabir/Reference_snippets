import pandas as pd
import numpy as np

def normalize_dataframes(ctrlData, testData):
    '''
    Takes two dataframes as input and returns the normalized versions for N number of columns.
    Parameters
    ----------
    ctrlData : Dataframe
        DESCRIPTION.
    testData : Dataframe
        DESCRIPTION.

    Returns
    -------
    Dataframe
    '''
    def get_numeric_columns(df):
        return df.select_dtypes(include=[np.number]).columns

    # Get numeric columns
    numeric_columns = get_numeric_columns(ctrlData)

    # Calculate means for each numeric column in the control dataframe
    means_ctrl = ctrlData[numeric_columns].mean()

    # Create new dataframes for normalized values
    ctrlData_normalized = pd.DataFrame()
    testData_normalized = pd.DataFrame()

    # Normalize each column
    for column in numeric_columns:
        mean_intensity = means_ctrl[column]
        ctrlData_normalized[f'{column}_Normalized'] = ctrlData[column] / mean_intensity
        testData_normalized[f'{column}_Normalized'] = testData[column] / mean_intensity

    # Combine original and normalized dataframes
    ctrlData_final = pd.concat([ctrlData, ctrlData_normalized], axis=1)
    testData_final = pd.concat([testData, testData_normalized], axis=1)

    return ctrlData_final, testData_final

ctrlData_normalized, testData_normalized = normalize_dataframes(ctrlData, testData)

####
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
