# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:51:28 2024

@author: kaabir
"""

import pandas as pd
import numpy as np

from scipy import stats
# https://towardsdatascience.com/the-ultimate-guide-to-finding-outliers-in-your-time-series-data-part-1-1bf81e09ade4

def grubbs_test(data, alpha=0.05):
    """
    Perform Grubbs' test for a single outlier.
    Parameters:
    - data: array-like, the dataset
    - alpha: significance level (default is 0.05)
    Returns:
    - outlier: True if a significant outlier is detected, otherwise False
    """
    n = len(data)
    if n < 3:
        raise ValueError("Grubbs' test requires at least 3 data points.")
    
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    
    # Calculate Grubbs' test statistic for the maximum and minimum value
    G_max = (np.max(data) - mean) / std_dev
    G_min = (mean - np.min(data)) / std_dev
    
    # Calculate the critical value for Grubbs' test
    critical_value = stats.t.ppf(1 - alpha / (2 * n), n - 2) * np.sqrt((n - 1) / np.sqrt(n - 2 + (stats.t.ppf(1 - alpha / (2 * n), n - 2))**2))
    
    # Determine if there's a significant outlier
    return G_max > critical_value or G_min > critical_value
    
def remove_Outliers(df, z_thresh=3):
    print("Outliers using Z-score method:")
    # Define a threshold for outlier detection
    """
    A z-score of 1 covers about 68% of the curve under a normal distribution, 
    A z-score of 2 covers about 95%, and 
    A z-score of 3 covers about 99.7%
    """
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):  # Process only numerical data
            df_col = df[col].dropna()  # Drop NaN values
            mean = df_col.mean()
            std_dev = df_col.std(ddof=0)
            z_scores = np.abs((df_col - mean) / std_dev)
            outliers = df_col[z_scores > z_thresh]
            print(f"{col} - Outliers: {len(outliers)}")
            print(outliers, "\n")

def remove_OutliersR(data, threshold=3.5):
    """
    Improves upon the traditional Z-score by using the median (M) instead of the mean, 
    as the mean is often not the most reliable statistical measure. 
    employs the median absolute deviation (MAD) rather than the standard deviation
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad
    return modified_z_scores

    
normalized_intens_dfs = []

def normalize_intensity(data_frame, mean_intensity_column, mean_intensity, folderName):
    if isinstance(data_frame, pd.DataFrame):
        normalized_column = data_frame[mean_intensity_column] / mean_intensity
        data_frame[f'{mean_intensity_column} Normalized {folderName}'] = normalized_column
    else:
        normalized_column = data_frame / mean_intensity

    normalized_df = pd.DataFrame({f'{mean_intensity_column} Normalized {folderName}': normalized_column})
    normalized_df.reset_index(drop=True, inplace=True)  # Reset the index
    normalized_intens_dfs.append(normalized_df)
    return data_frame  # Return the modified DataFrame or Series

normalized_dfs = []

def normalize_tomean_intensity(data1, data2, column1=None, column2=None):
    data1_values = data1.values
    data2_values = data2.values

    mean_intensity_col_0 = np.mean(data1_values)
    
    data1_normalized = data1_values / mean_intensity_col_0
    data2_normalized = data2_values / mean_intensity_col_0

    df1 = pd.DataFrame(data1_normalized, columns=[column1])
    df2 = pd.DataFrame(data2_normalized, columns=[column2])
    
    normalized_dfs.append(df1)
    normalized_dfs.append(df2)

    return df1, df2

"""
Non-Parametric T-Test

Mann-Whitney U Test for independent samples.

Wilcoxon Signed-Rank Test for paired samples.

Two-Way ANOVA: To analyze the effects of two factors and their interaction on a single dependent variable.

MANOVA: Extends ANOVA to multiple dependent variables to test the effects of factors.

"""

from scipy.stats import mannwhitneyu

def mann_whitney_u_test(group1, group2):
    """
    Perform Mann-Whitney U test for independent samples.
    
    Parameters:
    ----------
    group1 : array-like
    group2 : array-like
        
    Returns:
    -------
    tuple
        U statistic and p-value.
    """
    stat, p_value = mannwhitneyu(group1, group2)
    return stat, p_value

from scipy.stats import wilcoxon

def wilcoxon_signed_rank_test(AB1, AB2):
    """
    Perform Wilcoxon signed-rank test for paired samples.
    
    Parameters:
    ----------
    AB1 : array-like
    AB2 : array-like
        
    Returns:
    -------
    tuple
        Test statistic and p-value.
    """
    stat, p_value = wilcoxon(AB1, AB2)
    return stat, p_value

import statsmodels.api as sm
from statsmodels.formula.api import ols

def two_way_anova(data, dependent_var, factor1, factor2):
    """
    
    # Example data
    data = pd.DataFrame({
    'value': [20, 21, 19, 30, 31, 29, 40, 41, 39, 50, 51, 49],
    'factor1': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D'],
    'factor2': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z']
        })

    # Fit the model
        model = ols('value ~ C(factor1) * C(factor2)', data=data).fit()


    Perform Two-Way ANOVA.
    
    Parameters:
    ----------
    data : DataFrame
        Pandas DataFrame containing the data.
    dependent_var : str
        Name of the dependent variable.
    factor1 : str
        Name of the first factor.
    factor2 : str
        Name of the second factor.
        
    Returns:
    -------
    DataFrame
        ANOVA table.
    """
    model = ols(f'{dependent_var} ~ C({factor1}) * C({factor2})', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table


import pandas as pd
from statsmodels.multivariate.manova import MANOVA

def manova(data, dependent_vars, factor1, factor2):
    """
    # Example data
    data = pd.DataFrame({
        'dep1': [20, 21, 19, 30, 31, 29, 40, 41, 39, 50, 51, 49],
        'dep2': [22, 23, 21, 32, 33, 30, 42, 43, 41, 52, 53, 50],
        'factor1': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D'],
        'factor2': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z']
    })

    # Fit the MANOVA model
    manova = MANOVA.from_formula('dep1 + dep2 ~ C(factor1) * C(factor2)', data=data)    
    
    Perform MANOVA.
    
    Parameters:
    ----------
    data : DataFrame
        Pandas DataFrame containing the data.
    dependent_vars : list of str
        Names of the dependent variables.
    factor1 : str
        Name of the first factor.
    factor2 : str
        Name of the second factor.
        
    Returns:
    -------
    object
        MANOVA results.
    """
    formula = ' + '.join(dependent_vars) + ' ~ C(' + factor1 + ') * C(' + factor2 + ')'
    manova = MANOVA.from_formula(formula, data=data)
    results = manova.mv_test()
    return results
