# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:42:00 2023

@author: KurtV
"""

import numpy as np

#%% Utility Functions

#%%% Create censor/numeric columns

def split(df,
          value_column,
          censor_column = 'CensorComponent',
          numeric_column = 'NumericComponent'):
    '''
    A function that separates censors (<,≤,≥,>) from a result. Splitting the
    censor component from the numeric component enables the use of numeric
    data types. A result of <10 (string/text) is split into a censor component
    of < (string/text) and a numeric component of 10 (float/decimal).

    Parameters
    ----------
    df : DataFrame
        DataFrame containing a column to be split into components
    value_column : string
        Column name for the column that contains the results to be split
    censor_column : string, optional
        Column name that will be assigned to the censor component column
        The default is 'CensorComponent'.
    numeric_column : string, optional
        Column name that will be assigned to the numeric component column
        The default is 'NumericComponent'.

    Returns
    -------
    df : DataFrame
        Input DataFrame with two additional columns created for the censor
        and numeric components

    '''
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Ensure the value column is text/string data type.
    df[value_column] = df[value_column].astype(str)
    
    # Create censor and numeric columns from result column
    df[censor_column] = df[value_column].str.extract(r'([<≤≥>])').fillna('')
    df[numeric_column] = df[value_column].str.replace('<|≤|≥|>','',regex=True).astype(float)
    
    return df

#%%% Create interval for each result

def result_to_interval(df,
                       censor_column = 'CensorComponent',
                       numeric_column = 'NumericComponent',
                       include_negative_interval = False,
                       boundary_type = str):
    '''
    A function that utilises the censor and numeric components to convert
    each result into interval notation. Four columns are created to store
    the interval information including left/right bounds and whether the
    endpoints/boundaries are inclusive or exclusive (e.g., <1 vs ≤1)

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the censor and numeric columns
    censor_column : string, optional
        Column name for the column containing the censor component
        The default is 'CensorComponent'.
    numeric_column : string, optional
        Column name for the column containing the numeric component
        The default is 'NumericComponent'.
    include_negative_interval : boolean, optional
        If True, then all positive and negative values are considered
        (e.g., <0.5 would be converted to (-np.inf,5) if False).
        If False, then only non-negative values are considered
        (e.g., <0.5 would be converted to [0,5) if False).
        The default is False.
    boundary_type : DataType, optional
        This determines whether the boundary is returned in a string format
        (Open/Closed) or integer format (0 - open / 1 - closed).
        The default is str.

    Raises
    ------
    ValueError
        An error that is returned when results are found to be outside the
        assumed or defined lower and upper boundaries.

    Returns
    -------
    df : DataFrame
        Input DataFrame with four additional columns created that store the
        information for the intervals that represent the range of potential
        values for the results

    '''
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Define boundary values as string or integers
    closed_boundary = 'Closed'
    open_boundary = 'Open'
    # Use integers if type is set to int
    if boundary_type == int:
        closed_boundary = 1
        open_boundary = 0
    
    # Determine the lower bound
    if include_negative_interval:
        lower_bound = -np.inf
    else:
        lower_bound = 0.0
        # Raise error if negative values exist in data when negative intervals are excluded
        if (df['NumericComponent'] < 0.0).any():
            raise ValueError('Negative values exist in the data. Resolve negative'
                             'values or set include_negative_interval to True')
    
    # Determine conditions where the left boundary is open
    condition = (
        (include_negative_interval & df[censor_column].isin(['<','≤'])) |
        (df[censor_column] == '>')
        )
    df['LeftBoundary'] = np.where(condition, open_boundary, closed_boundary)
    
    # Determine conditions for where left bound is the lower_bound
    condition = (
        df[censor_column].isin(['<','≤'])
        )
    df['LeftBound'] = np.where(condition, lower_bound, df['NumericComponent'])
    
    # Determine conditions where right bound is infinite
    condition = (
        df[censor_column].isin(['≥','>'])
        )
    df['RightBound'] = np.where(condition, np.inf, df['NumericComponent'])
    
    # Determine conditions where the right bound is open
    condition = (
        df[censor_column].isin(['<','≥','>'])
        )
    df['RightBoundary'] = np.where(condition, open_boundary, closed_boundary)
    
    return df

def interval_notation(df):
    '''
    This function creates a column that combines the interval components
    into a single text notation for intervals.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing columns for left/right bounds and boundaries

    Returns
    -------
    df : DataFrame
        DataFrame identical to intput with additional column with combined
        interval notation

    '''
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Combine the left/right bound and boundary information into a single column
    df['Interval'] = np.where(df['LeftBoundary'].isin(['Open',0]), '(', '[') + \
                    df['LeftBound'].astype(str) + ', ' + df['RightBound'].astype(str) + \
                    np.where(df['RightBoundary'].isin(['Open', 0]), ')', ']')
    
    return df
    
    

#%%% Rounding method

def string_precision(value):
    '''
    A function that applies a specified rounding method that is value
    dependent. This method is specifically designed to reflect the
    typical measurement precision for water quality results. Depending on the
    value, the rounding is either to a particular number of decimal places or
    to a particular number of significant digits.

    Parameters
    ----------
    value : float
        Numeric value that will be rounded.

    Returns
    -------
    string : string
        The rounded value expressed as a string to the appropriate precision.

    '''
    
    # Values above 100 or are rounded to 100 should be expressed as integers
    # with no more than 3 significant digits.
    if round(value,1) >= 100:
        string = str(int(float(f'{value:.3g}')))
    # Values above 10 or are rounded to 10 should be rounded to 1 decimal place
    elif round(value,2) >= 10:
        string = f'{value:.1f}'
    # Values above 0.2 or are rounded to 0.2 should be rounded to 2 decimal places
    elif round(value,3) >= 0.2:
        string = f'{value:.2f}'
    # Values above 0.1 or are rounded to 0.1 should be rounded to 3 decimal places
    elif round(value,3) >= 0.1:
        string = f'{value:.3f}'
    # Values below 0.1 should be rounded to 2 significant digits
    else:
        string = f'{value:.2g}'

    return string


def numeric_precision(value):
    '''
    A function that returns a float data type from a rounding function instead
    of a string data type.

    Parameters
    ----------
    value : float
        Float value that may have more decimal places or significant digits
        than is appropriate

    Returns
    -------
    float
        Float value that is rounded appropriately

    '''
    
    # Return the same rounded result as string_precision but as a float
    return float(string_precision(value))

#%%% Create result from components

def create_result(df,
                  censor_column = 'CensorComponent',
                  numeric_column = 'NumericComponent',
                  precision_rounding = True):
    '''
    A function that combines censor and numeric components into a combined
    string/text result.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing censor and numeric columns that will be combined
    censor_column : string, optional
        Column name for the column containing the censor component
        The default is 'CensorComponent'.
    numeric_column : string, optional
        Column name for the column containing the numeric component
        The default is 'NumericComponent'.
    precision_rounding : boolean, optional
        If True, a rounding method is applied to round results to have no more
        decimals than what can be measured.
        The default is True.

    Returns
    -------
    df : DataFrame
        Input DataFrame that contains a new string/text column that combines
        the censor and numeric components of a result

    '''
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Combine the censor and numeric components to create a combined result
    # Apply the appropriate rounding functions if precision_rounding is True
    if precision_rounding:
        df['Result'] = df[censor_column] + df[numeric_column].apply(string_precision)
        df[numeric_column] = df[numeric_column].apply(numeric_precision)
    else:
        df['Result'] = df[censor_column] + df[numeric_column].astype(str)
    
    return df

#%% Maximum Result

def maximum_interval(df,
                     groupby_columns = []):
    '''
    A function that analyses the interval notation form of results returned
    from result_to_interval to generate a new interval for the maximum. Groups
    of results can be defined by including the columns that should be used to
    create groups.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains results in a specific interval notation.
    groupby_columns : list of strings, optional
        List of column names that should be used to create groups.
        The default is [].

    Returns
    -------
    df : DataFrame
        DataFrame that has the interval for the maximum (for each group if
        column names are provided for grouping)

    '''
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Create column that indicates the generated statistic and append to grouping list
    df['Statistic'] = 'Maximum'
    groupby_columns.append('Statistic')
    
    # Determine left bound and boundary for maximum for each group.
    
    # Consider the maximum bound for each left boundary option to determine
    # whether the bound should be open or closed.
    left = df.groupby(groupby_columns+['LeftBoundary'])['LeftBound'].max().unstack('LeftBoundary')
    # Use the maximum closed boundary if there are no open boundaries or
    # if the closed boundary is larger than the largest open boundary
    condition = (left['Closed'] > left['Open']) | (left['Open'].isna())
    left['LeftBoundary'] = np.where(condition,'Closed','Open')
    left['LeftBound'] = np.where(condition,left['Closed'],left['Open'])
    left = left[['LeftBoundary','LeftBound']]
    
    # Determine right bound and boundary for maximum for each group.
    
    # Consider the maximum bound for each right boundary option to determine
    # whether the bound should be open or closed.
    right = df.groupby(groupby_columns+['RightBoundary'])['RightBound'].max().unstack('RightBoundary')
    # Use the maximum closed boundary if there are no open boundaries or
    # if the closed boundary is larger than or equal to the largest open boundary
    condition = (right['Closed'] >= right['Open']) | (right['Open'].isna())
    right['RightBound'] = np.where(condition,right['Closed'],right['Open'])
    right['RightBoundary'] = np.where(condition,'Closed','Open')
    right = right[['RightBound','RightBoundary']]
    
    # Merge the two boundaries to create the interval for the maximum
    # Check that the merge is 1-to-1
    df = left.merge(right, how = 'outer', on = groupby_columns, validate = '1:1')
    
    return df
    

def maximum_result(df,
                   censor_column = 'CensorComponent',
                   numeric_column = 'NumericComponent',
                   precision_tolerance_to_drop_censor = 0.25):
    '''
    A function that determines the censor and numeric components for maxima
    from intervals that represent the maxima.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains intervals representing maxima
    censor_column : string, optional
        Column name to give the column containing the censor component
        The default is 'CensorComponent'.
    numeric_column : string, optional
        Column name to give the column containing the numeric component
        The default is 'NumericComponent'.
    precision_tolerance_to_drop_censor : float, optional
        Threshold for reporting censored vs non-censored results.
        Using the default, a maximum that is known to be between 0.3 and 0.5
        would be returned as 0.4, whereas a tolerance of 0 would yield a
        maximum of <0.5.
        The default is 0.25.

    Returns
    -------
    df : DataFrame
        DataFrame where intervals for maxima have been analysed to censor and
        numeric components.

    '''
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Determine the midpoint for the maximum interval if finite interval
    df['Midpoint'] = np.where((df['LeftBound'] > -np.inf) & (df['RightBound'] < np.inf),
                              0.5 * (df['LeftBound'] + df['RightBound']),
                              np.nan)
    
    # Define the potential conditions
    conditions = [
        # If the bounds are equal, then the result is uncensored
        (df['LeftBound'] == df['RightBound']),
        # If there is only a finite left bound, the result is right censored
        (df['RightBound'] == np.inf) & (df['LeftBound'] > -np.inf) & (df['LeftBoundary'] == 'Open'),
        (df['RightBound'] == np.inf) & (df['LeftBound'] > -np.inf) & (df['LeftBoundary'] == 'Closed'),
        # If the bounds are finite and the interval is within the precision tolerance, avoid the use of censors
        (df['RightBound'] - df['Midpoint']) <= df['Midpoint'] * precision_tolerance_to_drop_censor,
        # Otherwise, the maximum should be left censored
        (df['RightBound'] < np.inf) & (df['RightBoundary'] == 'Open'),
        (df['RightBound'] < np.inf) & (df['RightBoundary'] == 'Closed'),
        ]
    
    # Define the censor component for each condition
    censor_results = [
        '',
        '>',
        '≥',
        '',
        '<',
        '≤'
        ]
    
    # Define the numeric components for each condition
    numeric_results =[
        df['Midpoint'],
        df['LeftBound'],
        df['LeftBound'],
        df['Midpoint'],
        df['RightBound'],
        df['RightBound']
        ]
    
    # Determine the censor and numeric components
    # If no condition is met, default to <> and NaN
    df[censor_column] = np.select(conditions, censor_results, '<>')
    df[numeric_column] = np.select(conditions, numeric_results, np.nan)
    
    # Only return the censor and numeric columns
    df = df[['Interval',censor_column,numeric_column]]
    
    return df

def maximum(df,
            groupby_columns = [],
            values = ['CensorComponent','NumericComponent'],
            include_negative_interval = False,
            precision_tolerance_to_drop_censor = 0.25,
            precision_rounding = True):
    '''
    A function that combines the relevant maximum and utility functions to
    generate the maxima results for groups within a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains censored or uncensored results
    groupby_columns : list of strings, optional
        List of column names that should be used to create groups. A maximum
        will be found within each group.
        The default is [].
    values : list of strings, optional
        The column name(s) for the column(s) that contain the results. If a 
        single column name is given, it is assumed that the column contains
        combined censor and numeric components. If two column names are
        provided, then the first should only contain one of five censors (<,≤,≥,>)
        and the second should contain only numeric data.
        The default is ['CensorComponent','NumericComponent'].
    include_negative_interval : boolean, optional
        If True, then all positive and negative values are considered
        (e.g., <0.5 would be converted to (-np.inf,5) if False).
        If False, then only non-negative values are considered
        (e.g., <0.5 would be converted to [0,5) if False).
        The default is False.
    precision_tolerance_to_drop_censor : float, optional
        Threshold for reporting censored vs non-censored results.
        Using the default, a maximum that is known to be between 0.3 and 0.5
        would be returned as 0.4, whereas a tolerance of 0 would yield a
        maximum of <0.5.
        The default is 0.25.
    precision_rounding : boolean, optional
        If True, a rounding method is applied to round results to have no more
        decimals than what can be measured.
        The default is True.

    Returns
    -------
    df : DataFrame
        DataFrame that contains a column with the relevant maximum or maxima

    '''
    
    # If single result column provided, then split column
    if len(values) == 1:
        censor_column = 'CensorComponent'
        numeric_column = 'NumericComponent'
        df = split(df,values[0])
    # Else define the names to use for the censor and numeric columns
    else:
        censor_column = values[0]
        numeric_column = values[1]
    
    # Convert the results from censor and numeric components to an interval representation
    df = result_to_interval(df, censor_column, numeric_column, include_negative_interval)
    
    # Using the intervals, determine the range of possible maxima
    df = maximum_interval(df, groupby_columns)
    
    # Create interval notation for the maximum
    df = interval_notation(df)
    
    # Convert the interval for the maximum into censor and numeric notation
    df = maximum_result(df, censor_column, numeric_column, precision_tolerance_to_drop_censor)
    
    # Combine the censor and numeric components into a result
    df = create_result(df, censor_column, numeric_column, precision_rounding)
    
    return df

#%% Minimum Result

def minimum_interval(df,
                     groupby_columns = []):
    '''
    A function that analyses the interval notation form of results returned
    from result_to_interval to generate a new interval for the minimum. Groups
    of results can be defined by including the columns that should be used to
    create groups.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains results in a specific interval notation.
    groupby_columns : list of strings, optional
        List of column names that should be used to create groups.
        The default is [].

    Returns
    -------
    df : DataFrame
        DataFrame that has the interval for the minimum (for each group if
        column names are provided for grouping)

    '''
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Create column that indicates the generated statistic and append to grouping list
    df['Statistic'] = 'Minimum'
    groupby_columns.append('Statistic')
    
    # Determine left bound and boundary for minimum for each group.
    
    # Consider the minimum bound for each left boundary option to determine
    # whether the bound should be open or closed.
    left = df.groupby(groupby_columns+['LeftBoundary'])['LeftBound'].min().unstack('LeftBoundary')
    # Use the minimum closed boundary if there are no open boundaries or
    # if the closed boundary is less than or equal to the least open boundary
    condition = (left['Closed'] <= left['Open']) | (left['Open'].isna())
    left['LeftBoundary'] = np.where(condition,'Closed','Open')
    left['LeftBound'] = np.where(condition,left['Closed'],left['Open'])
    left = left[['LeftBoundary','LeftBound']]
    
    # Determine right bound and boundary for minimum for each group.
    
    # Consider the minimum bound for each right boundary option to determine
    # whether the bound should be open or closed.
    right = df.groupby(groupby_columns+['RightBoundary'])['RightBound'].min().unstack('RightBoundary')
    # Use the minimum closed boundary if there are no open boundaries or
    # if the closed boundary is less than the least open boundary
    condition = (right['Closed'] < right['Open']) | (right['Open'].isna())
    right['RightBound'] = np.where(condition,right['Closed'],right['Open'])
    right['RightBoundary'] = np.where(condition,'Closed','Open')
    right = right[['RightBound','RightBoundary']]
    
    # Merge the two boundaries to create the interval for the minimum
    # Check that the merge is 1-to-1
    df = left.merge(right, how = 'outer', on = groupby_columns, validate = '1:1')
    
    return df
    

def minimum_result(df,
                   censor_column = 'CensorComponent',
                   numeric_column = 'NumericComponent',
                   include_negative_interval = False,
                   precision_tolerance_to_drop_censor = 0.25):
    '''
    A function that determines the censor and numeric components for minima
    from intervals that represent the minima.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains intervals representing minima
    censor_column : string, optional
        Column name to give the column containing the censor component
        The default is 'CensorComponent'.
    numeric_column : string, optional
        Column name to give the column containing the numeric component
        The default is 'NumericComponent'.
    include_negative_interval : boolean, optional
        If True, then all positive and negative values are considered
        (e.g., <0.5 would be converted to (-np.inf,5) if False).
        If False, then only non-negative values are considered
        (e.g., <0.5 would be converted to [0,5) if False).
        The default is False.
    precision_tolerance_to_drop_censor : float, optional
        Threshold for reporting censored vs non-censored results.
        Using the default, a minimum that is known to be between 0.3 and 0.5
        would be returned as 0.4, whereas a tolerance of 0 would yield a
        minimum of >0.3.
        The default is 0.25.

    Returns
    -------
    df : DataFrame
        DataFrame where intervals for minima have been analysed to censor and
        numeric components.

    '''
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Determine the midpoint for the minimum interval if finite interval
    df['Midpoint'] = np.where((df['LeftBound'] > -np.inf) & (df['RightBound'] < np.inf),
                              0.5 * (df['LeftBound'] + df['RightBound']),
                              np.nan)
    
    # Determine the lower bound
    if include_negative_interval:
        lower_bound = -np.inf
    else:
        lower_bound = 0.0
    
    # Define the potential conditions
    conditions = [
        # If the bounds are equal, then the result is uncensored
        (df['LeftBound'] == df['RightBound']),
        # If there is only a finite right bound, the result is left censored
        (df['LeftBound'] == lower_bound) & (df['RightBound'] < np.inf) & (df['RightBoundary'] == 'Open'),
        (df['LeftBound'] == lower_bound) & (df['RightBound'] < np.inf) & (df['RightBoundary'] == 'Closed'),
        # If the bounds are finite and the interval is within the precision tolerance, avoid the use of censors
        (df['RightBound'] - df['Midpoint']) <= df['Midpoint'] * precision_tolerance_to_drop_censor,
        # Otherwise, the minimum should be right censored
        (df['LeftBound'] > lower_bound) & (df['LeftBoundary'] == 'Open'),
        (df['LeftBound'] > lower_bound) & (df['LeftBoundary'] == 'Closed'),
        ]
    
    # Define the censor component for each condition
    censor_results = [
        '',
        '<',
        '≤',
        '',
        '>',
        '≥'
        ]
    
    # Define the numeric components for each condition
    numeric_results =[
        df['Midpoint'],
        df['RightBound'],
        df['RightBound'],
        df['Midpoint'],
        df['LeftBound'],
        df['LeftBound']
        ]
    
    # Determine the censor and numeric components
    # If no condition is met, default to <> and NaN
    df[censor_column] = np.select(conditions, censor_results, '<>')
    df[numeric_column] = np.select(conditions, numeric_results, np.nan)
    
    # Only return the censor and numeric columns
    df = df[['Interval',censor_column,numeric_column]]
    
    return df

def minimum(df,
            groupby_columns = [],
            values = ['CensorComponent','NumericComponent'],
            include_negative_interval = False,
            precision_tolerance_to_drop_censor = 0.25,
            precision_rounding = True):
    '''
    A function that combines the relevant minimum and utility functions to
    generate the minima results for groups within a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains censored or uncensored results
    groupby_columns : list of strings, optional
        List of column names that should be used to create groups. A minimum
        will be found within each group.
        The default is [].
    values : list of strings, optional
        The column name(s) for the column(s) that contain the results. If a 
        single column name is given, it is assumed that the column contains
        combined censor and numeric components. If two column names are
        provided, then the first should only contain one of five censors (<,≤,≥,>)
        and the second should contain only numeric data.
        The default is ['CensorComponent','NumericComponent'].
    include_negative_interval : boolean, optional
        If True, then all positive and negative values are considered
        (e.g., <0.5 would be converted to (-np.inf,5) if False).
        If False, then only non-negative values are considered
        (e.g., <0.5 would be converted to [0,5) if False).
        The default is False.
    precision_tolerance_to_drop_censor : float, optional
        Threshold for reporting censored vs non-censored results.
        Using the default, a minimum that is known to be between 0.3 and 0.5
        would be returned as 0.4, whereas a tolerance of 0 would yield a
        minimum of >0.3.
        The default is 0.25.
    precision_rounding : boolean, optional
        If True, a rounding method is applied to round results to have no more
        decimals than what can be measured.
        The default is True.

    Returns
    -------
    df : DataFrame
        DataFrame that contains a column with the relevant minimum or minima

    '''
    
    # If single result column provided, then split column
    if len(values) == 1:
        censor_column = 'CensorComponent'
        numeric_column = 'NumericComponent'
        df = split(df,values[0])
    # Else define the names to use for the censor and numeric columns
    else:
        censor_column = values[0]
        numeric_column = values[1]
    
    # Convert the results from censor and numeric components to an interval representation
    df = result_to_interval(df, censor_column, numeric_column, include_negative_interval)
    
    # Using the intervals, determine the range of possible minima
    df = minimum_interval(df, groupby_columns)
    
    # Create interval notation for the minimum
    df = interval_notation(df)
    
    # Convert the interval for the minimum into censor and numeric notation
    df = minimum_result(df, censor_column, numeric_column, include_negative_interval, precision_tolerance_to_drop_censor)
    
    # Combine the censor and numeric components into a result
    df = create_result(df, censor_column, numeric_column, precision_rounding)
    
    return df