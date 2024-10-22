import pandas as pd
import numpy as np

def normalize_multiple_dataframes(*dataframes, names):
    '''
        Takes Nth dataframes as input and returns the normalized versions for Nth number of columns.
        This divides from the mean of Control assumed to placed at the first column
        Parameters
        ----------
        Returns
        -------
        Dataframe
    '''
    def get_numeric_columns(df):
        return df.select_dtypes(include=[np.number]).columns

    # Assume the first dataframe is the control
    control_df = dataframes[0]
    numeric_columns = get_numeric_columns(control_df)

    # Calculate means for each numeric column in the control dataframe
    control_means = control_df[numeric_columns].mean()

    normalized_dataframes = {}

    for df, name in zip(dataframes, names):
        df_normalized = pd.DataFrame(index=df.index)
        for column in numeric_columns:
            if column in df.columns:
                mean_intensity = control_means[column]
                if mean_intensity != 0:  # Avoid division by zero
                    df_normalized[f'{column}_Normalized'] = df[column] / mean_intensity
                else:
                    df_normalized[f'{column}_Normalized'] = df[column]  # exception to handle
        
        # Combine original and normalized dataframes
        df_final = pd.concat([df, df_normalized], axis=1)
        normalized_dataframes[name] = df_final

    return normalized_dataframes
        
def normalize_multiple_dataframes(*dataframes, names):
        '''
    Takes Nth dataframes as input and returns the normalized versions for Nth number of columns independant of any column
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

    # Get all unique numeric columns across all dataframes
    all_numeric_columns = set()
    for df in dataframes:
        all_numeric_columns.update(get_numeric_columns(df))

    # Calculate means for each numeric column across all dataframes
    all_means = pd.Series(dtype=float)
    for column in all_numeric_columns:
        column_data = [df[column] for df in dataframes if column in df.columns]
        if column_data:
            all_means[column] = pd.concat(column_data).mean()

    normalized_dataframes = {}

    for df, name in zip(dataframes, names):
        df_normalized = pd.DataFrame(index=df.index)
        for column in all_numeric_columns:
            if column in df.columns and column in all_means.index:
                mean = all_means[column]
                df_normalized[f'{column}_Normalized'] = df[column] / mean
        
        # Combine original and normalized dataframes
        df_final = pd.concat([df, df_normalized], axis=1)
        normalized_dataframes[name] = df_final

    return normalized_dataframes

# Usage 
normalized_dfs = normalize_multiple_dataframes(Ctrl, test1, 
                                               test2, test3,
                                               test4, 
                                               names=['Ctrl', 'test1',
                                                      'test2','test3',
                                                      'test3',
                                                      ])
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
        ctrlData_normalized[f'{column}_Normalized'] = ctrlData[column] / mean
        testData_normalized[f'{column}_Normalized'] = testData[column] / mean_

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
