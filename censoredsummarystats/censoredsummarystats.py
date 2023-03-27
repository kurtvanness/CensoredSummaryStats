# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:42:00 2023

@author: KurtV
"""

import numpy as np

#%% Utility Functions

#%%% Display

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
    
    # Check for infinite values
    if abs(value) == np.inf:
        string = str(value)
    # Values above 100 or are rounded to 100 should be expressed as integers
    # with no more than 3 significant digits.
    elif round(abs(value),1) >= 100:
        string = str(int(float(f'{value:.3g}')))
    # Values above 10 or are rounded to 10 should be rounded to 1 decimal place
    elif round(abs(value),2) >= 10:
        string = f'{value:.1f}'
    # Values above 0.2 or are rounded to 0.2 should be rounded to 2 decimal places
    elif round(abs(value),3) >= 0.2:
        string = f'{value:.2f}'
    # Values above 0.1 or are rounded to 0.1 should be rounded to 3 decimal places
    elif round(abs(value),3) >= 0.1:
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

def interval_notation(df, precision_rounding=True):
    '''
    This function creates a column that combines the interval components
    into a single text notation for intervals.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing columns for left/right bounds and boundaries
    precision_rounding : boolean, optional
        If True, a rounding method is applied to round results to have no more
        decimals than what can be measured.
        The default is True.

    Returns
    -------
    df : DataFrame
        DataFrame identical to intput with additional column with combined
        interval notation

    '''
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Determine the left boundary symbol
    df['Interval'] = np.where(df['LeftBoundary'] == 'Open', '(', '[')
    
    # Incorporate left and right bounds
    if precision_rounding:
        df['Interval'] += df['LeftBound'].apply(string_precision) + ', ' + df['RightBound'].apply(string_precision)
    else:
        df['Interval'] += df['LeftBound'].astype(str) + ', ' + df['RightBound'].astype(str)
    
    # Determine the right boundary symbol
    df['Interval'] += np.where(df['RightBoundary'].isin(['Open']), ')', ']')
    
    return df

#%%% Convert between, results, components, intervals

def result_to_components(df,
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

def components_to_interval(df,
                           censor_column = 'CensorComponent',
                           numeric_column = 'NumericComponent',
                           include_negative_interval = False):
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
        e.g., <0.5 would be converted to (-np.inf,5).
        If False, then only non-negative values are considered
        e.g., <0.5 would be converted to [0,5).
        The default is False.
    
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
    
    # Define where the left bound is closed
    df['LeftBoundary'] = np.where(df[censor_column].isin(['','≥']),
                                  closed_boundary, open_boundary)
    
    # Define where left bound is unlimited
    df['LeftBound'] = np.where(df[censor_column].isin(['<','≤']),
                               -np.inf, df[numeric_column])
    
    # Define where right bound is unlimited
    df['RightBound'] = np.where(df[censor_column].isin(['≥','>']),
                                np.inf, df[numeric_column])
    
    # Define where the right bound is closed
    df['RightBoundary'] = np.where(df[censor_column].isin(['≤','']),
                                   closed_boundary, open_boundary)
    
    if not include_negative_interval:
        if df['LeftBound'].between(-np.inf, 0, inclusive='neither').any():
            raise ValueError('Negative values exist in the data. Resolve negative'
                             'values or set include_negative_interval to True')
        
        condition = (df['LeftBound'] < 0)
        df['LeftBoundary'] = np.where(condition, 'Closed', df['LeftBoundary'])
        df['LeftBound'] = np.where(condition, 0.0, df['LeftBound'])
    
    return df


def interval_to_components(df,
                       censor_column = 'CensorComponent',
                       numeric_column = 'NumericComponent',
                       focus_high_potential = True,
                       include_negative_interval = False,
                       precision_tolerance_to_drop_censor = 0.25):
    '''
    A function that determines the censor and numeric components
    from intervals

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains interval information
    censor_column : string, optional
        Column name to give the column containing the censor component
        The default is 'CensorComponent'.
    numeric_column : string, optional
        Column name to give the column containing the numeric component
        The default is 'NumericComponent'.
    focus_high_potential : boolean, optional
        If True, then information on the highest potential result will be
        focused over the lowest potential result.
    include_negative_interval : boolean, optional
        If True, then all positive and negative values are considered
        e.g., <0.5 would be converted to (-np.inf,5).
        If False, then only non-negative values are considered
        e.g., <0.5 would be converted to [0,5).
        This setting only affects results if focus_high_potential is False.
        The default is False.
    precision_tolerance_to_drop_censor : float, optional
        Threshold for reporting censored vs non-censored results.
        Using the default, a result that is known to be in the interval (0.3, 0.5)
        would be returned as 0.4, whereas a tolerance of 0 would yield a
        result of <0.5 or >0.3 depending on the value of focus_highest_potential.
        The default is 0.25.

    Returns
    -------
    df : DataFrame
        DataFrame where intervals for results have been analysed to censor and
        numeric components.

    '''
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Determine the midpoint for the interval if finite interval
    df['Midpoint'] = np.where((df['LeftBound'] > -np.inf) & (df['RightBound'] < np.inf),
                              0.5 * (df['LeftBound'] + df['RightBound']),
                              np.nan)
    
    
    # Define the conditions for an uncensored result
    conditions = [
        # If the bounds are equal, then the result is uncensored
        (df['LeftBound'] == df['RightBound']),
        # If the bounds are finite and the interval is within the precision tolerance, avoid the use of censors
        (df['RightBound'] - df['Midpoint']) <= df['Midpoint'] * precision_tolerance_to_drop_censor
        ]
    
    censor_results = [
        '',
        ''
        ]
    
    numeric_results =[
        df['Midpoint'],
        df['Midpoint']
        ]
    
    # If focused on the highest potential result
    if focus_high_potential:
        # Set censor and numeric components for each condition
        conditions += [
            # If there is an infinite right bound, the result is right censored
            (df['RightBound'] == np.inf) & (df['LeftBound'] > -np.inf) & (df['LeftBoundary'] == 'Open'),
            (df['RightBound'] == np.inf) & (df['LeftBound'] > -np.inf) & (df['LeftBoundary'] == 'Closed'),
            # Otherwise, the result should be left censored
            (df['RightBound'] < np.inf) & (df['RightBoundary'] == 'Open'),
            (df['RightBound'] < np.inf) & (df['RightBoundary'] == 'Closed')
            ]
        
        censor_results += [
            '>',
            '≥',
            '<',
            '≤'
            ]
        
        numeric_results += [
            df['LeftBound'],
            df['LeftBound'],
            df['RightBound'],
            df['RightBound']
            ]
    # Else focused on the lowest potential result
    else:
        # Determine the lower bound
        if include_negative_interval:
            lower_bound = -np.inf
        else:
            lower_bound = 0.0
        # Set censor and numeric components for each condition
        conditions += [
            # If the left bound is identical to the lowest potential lower bound,
            # then the result is left censored
            (df['LeftBound'] == lower_bound) & (df['RightBound'] < np.inf) & (df['RightBoundary'] == 'Open'),
            (df['LeftBound'] == lower_bound) & (df['RightBound'] < np.inf) & (df['RightBoundary'] == 'Closed'),
            # Otherwise, the result should be right censored
            (df['LeftBound'] > lower_bound) & (df['LeftBoundary'] == 'Open'),
            (df['LeftBound'] > lower_bound) & (df['LeftBoundary'] == 'Closed')
            ]
        
        censor_results += [
            '<',
            '≤',
            '>',
            '≥'
            ]
        
        numeric_results += [
            df['RightBound'],
            df['RightBound'],
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


def components_to_result(df,
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
    from components_to_interval to generate a new interval for the maximum. Groups
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
    df = df.copy().reset_index()
    
    # Create column that indicates the generated statistic and append to grouping list
    df['Statistic'] = 'Maximum'
    groupby_columns.append('Statistic')
    
    # Determine left bound and boundary for maximum for each group.
    
    # Consider the maximum bound for each left boundary option to determine
    # whether the bound should be open or closed.
    left = df.groupby(groupby_columns+['LeftBoundary'])['LeftBound'].max().unstack('LeftBoundary')
    # Create missing columns
    for item in ['Open','Closed']:
        if item not in left.columns:
            left[item] = np.nan
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
    # Create missing columns
    for item in ['Open','Closed']:
        if item not in right.columns:
            right[item] = np.nan
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

def maximum(df,
            groupby_columns = [[]],
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
    groupby_columns : list of lists of strings, optional
        List of column names that should be used to create groups. A maximum
        will be found within each group. Multiple lists can be supplied to
        perform sequential maxima before converting intervals to a result.
        The default is [[]].
    values : list of strings, optional
        The column name(s) for the column(s) that contain the results. If a 
        single column name is given, it is assumed that the column contains
        combined censor and numeric components. If two column names are
        provided, then the first should only contain one of five censors (<,≤,≥,>)
        and the second should contain only numeric data.
        The default is ['CensorComponent','NumericComponent'].
    include_negative_interval : boolean, optional
        If True, then all positive and negative values are considered
        e.g., <0.5 would be converted to (-np.inf,5).
        If False, then only non-negative values are considered
        e.g., <0.5 would be converted to [0,5).
        This setting only affects results if focus_high_potential is False.
        The default is False.
    precision_tolerance_to_drop_censor : float, optional
        Threshold for reporting censored vs non-censored results.
        Using the default, a result that is known to be in the interval (0.3, 0.5)
        would be returned as 0.4, whereas a tolerance of 0 would yield a
        result of <0.5 or >0.3 depending on the value of focus_highest_potential.
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
        df = result_to_components(df,values[0])
    # Else define the names to use for the censor and numeric columns
    else:
        censor_column = values[0]
        numeric_column = values[1]
    
    # Convert the results from censor and numeric components to an interval representation
    df = components_to_interval(df, censor_column, numeric_column, include_negative_interval)
    
    # Cycle through each grouping
    for grouping in groupby_columns:
        # Using the intervals, determine the range of possible maxima
        df = maximum_interval(df, grouping)
    
    # Create interval notation for the maximum
    df = interval_notation(df, precision_rounding)
    
    # Convert the interval for the maximum into censor and numeric notation
    df = interval_to_components(df, censor_column, numeric_column,
                                focus_high_potential = True,
                                include_negative_interval = include_negative_interval,
                                precision_tolerance_to_drop_censor = precision_tolerance_to_drop_censor)
    
    # Combine the censor and numeric components into a result
    df = components_to_result(df, censor_column, numeric_column, precision_rounding)
    
    return df

#%% Minimum Result

def minimum_interval(df,
                     groupby_columns = []):
    '''
    A function that analyses the interval notation form of results returned
    from components_to_interval to generate a new interval for the minimum. Groups
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
    df = df.copy().reset_index()
    
    # Create column that indicates the generated statistic and append to grouping list
    df['Statistic'] = 'Minimum'
    groupby_columns.append('Statistic')
    
    # Determine left bound and boundary for minimum for each group.
    
    # Consider the minimum bound for each left boundary option to determine
    # whether the bound should be open or closed.
    left = df.groupby(groupby_columns+['LeftBoundary'])['LeftBound'].min().unstack('LeftBoundary')
    # Create missing columns
    for item in ['Open','Closed']:
        if item not in left.columns:
            left[item] = np.nan
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
    # Create missing columns
    for item in ['Open','Closed']:
        if item not in right.columns:
            right[item] = np.nan
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

def minimum(df,
            groupby_columns = [[]],
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
    groupby_columns : list of lists of strings, optional
        List of column names that should be used to create groups. A minimum
        will be found within each group. Multiple lists can be supplied to
        perform sequential minima before converting intervals to a result.
        The default is [[]].
    values : list of strings, optional
        The column name(s) for the column(s) that contain the results. If a 
        single column name is given, it is assumed that the column contains
        combined censor and numeric components. If two column names are
        provided, then the first should only contain one of five censors (<,≤,≥,>)
        and the second should contain only numeric data.
        The default is ['CensorComponent','NumericComponent'].
    include_negative_interval : boolean, optional
        If True, then all positive and negative values are considered
        e.g., <0.5 would be converted to (-np.inf,5).
        If False, then only non-negative values are considered
        e.g., <0.5 would be converted to [0,5).
        This setting only affects results if focus_high_potential is False.
        The default is False.
    precision_tolerance_to_drop_censor : float, optional
        Threshold for reporting censored vs non-censored results.
        Using the default, a result that is known to be in the interval (0.3, 0.5)
        would be returned as 0.4, whereas a tolerance of 0 would yield a
        result of <0.5 or >0.3 depending on the value of focus_highest_potential.
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
        df = result_to_components(df,values[0])
    # Else define the names to use for the censor and numeric columns
    else:
        censor_column = values[0]
        numeric_column = values[1]
    
    # Convert the results from censor and numeric components to an interval representation
    df = components_to_interval(df, censor_column, numeric_column, include_negative_interval)
    
    # Cycle through each grouping
    for grouping in groupby_columns:
        # Using the intervals, determine the range of possible minima
        df = minimum_interval(df, grouping)
    
    # Create interval notation for the minimum
    df = interval_notation(df, precision_rounding)
    
    # Convert the interval for the minimum into censor and numeric notation
    df = interval_to_components(df, censor_column, numeric_column,
                                focus_high_potential = False,
                                include_negative_interval = include_negative_interval,
                                precision_tolerance_to_drop_censor = precision_tolerance_to_drop_censor)
    
    # Combine the censor and numeric components into a result
    df = components_to_result(df, censor_column, numeric_column, precision_rounding)
    
    return df

#%% Average Result

def average_interval(df,
                     groupby_columns = []):
    '''
    A function that analyses the interval notation form of results returned
    from components_to_interval to generate a new interval for the average. Groups
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
        DataFrame that has the interval for the average (for each group if
        column names are provided for grouping)

    '''
    
    # Create a copy of the DataFrame
    df = df.copy().reset_index()
    
    # Create column that indicates the generated statistic and append to grouping list
    df['Statistic'] = 'Average'
    groupby_columns.append('Statistic')
    
    # Change notation of 'Closed' and 'Open' boundaries to 0 and 1, respectively
    # The presence of any open boundaries on one side ensure that the interval for the average is also
    # open on that side
    df[['LeftBoundary','RightBoundary']] = df[['LeftBoundary','RightBoundary']].replace(['Closed','Open'], [0,1])
    
    # Get the left/right bounds of the average by averaging bounds within the group
    # Determine whether any Open (now value of 1) boundaries exist. If there are
    # any open boundaries used in the average, then the resulting average will be open
    df = df.groupby(groupby_columns).agg(LeftBoundary = ('LeftBoundary', 'max'),
                                         Minimum = ('LeftBound','min'),
                                         LeftBound = ('LeftBound','mean'),
                                         RightBound = ('RightBound','mean'),
                                         Maximum = ('RightBound','max'),
                                         RightBoundary = ('RightBoundary','max'))
    
    # Replace integers with text for boundaries
    df[['LeftBoundary','RightBoundary']] = df[['LeftBoundary','RightBoundary']].replace([0,1], ['Closed','Open'])
    
    # Means with infinite values produce nan values rather than np.inf values
    # Convert nan to inf only if infinite values are included in the mean
    df['LeftBound'] = np.where(df['Minimum'] == -np.inf, -np.inf, df['LeftBound'])
    df['RightBound'] = np.where(df['Maximum'] == np.inf, np.inf, df['RightBound'])
    
    return df

def average(df,
            groupby_columns = [[]],
            values = ['CensorComponent','NumericComponent'],
            focus_high_potential = True,
            include_negative_interval = False,
            precision_tolerance_to_drop_censor = 0.25,
            precision_rounding = True):
    '''
    A function that combines the relevant average and utility functions to
    generate the average results for groups within a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains censored or uncensored results
    groupby_columns : list of lists of strings, optional
        List of column names that should be used to create groups. An average
        will be found within each group. Multiple lists can be supplied to
        perform sequential averaging before converting intervals to a result.
        The default is [[]].
    values : list of strings, optional
        The column name(s) for the column(s) that contain the results. If a 
        single column name is given, it is assumed that the column contains
        combined censor and numeric components. If two column names are
        provided, then the first should only contain one of five censors (<,≤,≥,>)
        and the second should contain only numeric data.
        The default is ['CensorComponent','NumericComponent'].
    focus_high_potential : boolean, optional
        If True, then information on the highest potential result will be
        focused over the lowest potential result.
    include_negative_interval : boolean, optional
        If True, then all positive and negative values are considered
        e.g., <0.5 would be converted to (-np.inf,5).
        If False, then only non-negative values are considered
        e.g., <0.5 would be converted to [0,5).
        This setting only affects results if focus_high_potential is False.
        The default is False.
    precision_tolerance_to_drop_censor : float, optional
        Threshold for reporting censored vs non-censored results.
        Using the default, a result that is known to be in the interval (0.3, 0.5)
        would be returned as 0.4, whereas a tolerance of 0 would yield a
        result of <0.5 or >0.3 depending on the value of focus_highest_potential.
        The default is 0.25.
    precision_rounding : boolean, optional
        If True, a rounding method is applied to round results to have no more
        decimals than what can be measured.
        The default is True.

    Returns
    -------
    df : DataFrame
        DataFrame that contains calculated averages

    '''
    
    # If single result column provided, then split column
    if len(values) == 1:
        censor_column = 'CensorComponent'
        numeric_column = 'NumericComponent'
        df = result_to_components(df,values[0])
    # Else define the names to use for the censor and numeric columns
    else:
        censor_column = values[0]
        numeric_column = values[1]
    
    # Convert the results from censor and numeric components to an interval representation
    df = components_to_interval(df, censor_column, numeric_column, include_negative_interval)
    
    # Cycle through each grouping
    for grouping in groupby_columns:
        # Using the intervals, determine the range of possible averages
        df = average_interval(df, grouping)
    
    # Create interval notation for the minimum
    df = interval_notation(df, precision_rounding)
    
    # Convert the interval for the minimum into censor and numeric notation
    df = interval_to_components(df, censor_column, numeric_column,
                                focus_high_potential = focus_high_potential,
                                include_negative_interval = include_negative_interval,
                                precision_tolerance_to_drop_censor = precision_tolerance_to_drop_censor)
    
    # Combine the censor and numeric components into a result
    df = components_to_result(df, censor_column, numeric_column, precision_rounding)
    
    return df

#%% Percentile Result

def percentile_interval(df,
                        percentile,
                        method = 'hazen',
                        groupby_columns = []):
    '''
    A function that analyses the interval notation form of results returned
    from components_to_interval to generate a new interval for percentiles. Groups
    of results can be defined by including the columns that should be used to
    create groups.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains results in a specific interval notation.
    percentile : float
        The desired percentile. Values should be between 0 and 100.
    method : string
        The percentile method. The options and definitions come from:
            https://environment.govt.nz/assets/Publications/Files/hazen-percentile-calculator-2.xls
            Options include the following, ordered from largest to smallest result.
            - weiball
            - tukey
            - blom
            - hazen
            - excel
        The default is hazen.
    groupby_columns : list of strings, optional
        List of column names that should be used to create groups.
        The default is [].

    Returns
    -------
    df : DataFrame
        DataFrame that has the interval for the percentile (for each group if
        column names are provided for grouping)

    '''
    
    # Create a copy of the DataFrame
    df = df.copy().reset_index()
    
    # Check the percentile is between 0 and 1
    if percentile < 0 or percentile > 100:
        raise Exception(f'Percentile out of range. Attempted percentile: {percentile}')
    
    # Create column that indicates the generated statistic and append to grouping list
    df['Statistic'] = f'Percentile-{percentile}'
    groupby_columns.append('Statistic')
    
    # Convert percentile to be between 0 and 1
    percentile = percentile/100
    
    # Determine size of each group
    df['Size'] = df.groupby(groupby_columns)['SiteID'].transform('size')
    
    # Set values for percentile methods
    method_dict = {'weiball':0.0, 'tukey':1/3, 'blom':3/8, 'hazen':1/2, 'excel':1.0}
    # https://en.wikipedia.org/wiki/Percentile
    C = method_dict[method]
    
    # Ensure rank is at least 1 and no more than len(data)
    minimum_size = round(C + (1-C)*max((1-percentile)/percentile,
                                       percentile/(1-percentile)),10)
    
    # Only consider groups that meet the minimum size requirement
    df = df[df['Size'] >= minimum_size]
    
    # Determine left bound and boundary for each group
    
    # Change notation of 'Closed' and 'Open' boundaries to 0 and 1, respectively
    # Use 0 for closed to ensure that closed boundaries are sorted less than open
    # boundaries when the left bound is tied
    df['LeftBoundary'] = df['LeftBoundary'].replace(['Closed','Open'], [0,1])
    
    # Sort left bound values
    left = df.copy()[groupby_columns+['Size','LeftBoundary','LeftBound']].sort_values(by=['LeftBound','LeftBoundary'])
    
    # Add index for each group
    left['Index'] = left.groupby(groupby_columns).cumcount() + 1
    
    # Determine the rank in each group for the percentile
    left['Rank'] = round(C + percentile*(left['Size'] + 1 - 2*C),8)
    
    # Generate proximity of each result to percentile rank using the index
    left['Proximity'] = 0
    
    conditions = [
        # If the percentile rank is a whole number, then use that index result
        (left['Rank'] == left['Index']),
        # If the percentile rank is less than 1 above the index value,
        # then assign the appropriate contribution to that index value
        (left['Rank'] - left['Index']).between(0,1,inclusive='neither'),
        # If the percentile rank is less than 1 below the index value,
        # then assign the appropriate contribution to that index value
        (left['Index'] - left['Rank']).between(0,1,inclusive='neither')
        ]
    
    results = [
        1,
        1 - (left['Rank'] - left['Index']),
        1 - (left['Index'] - left['Rank']),
        ]
    
    left['Proximity'] = np.select(conditions, results, np.nan)
    
    # Drop non-contributing rows
    left = left[left['Proximity'] > 0]
    
    # Calculate contribution for index values that contribute to the result
    left['Contribution'] = left['Proximity'] * left['LeftBound']
    
    # Determine left bound and boundary using the sum of the contributions
    # and an open boundary if any of the contributing values is open
    left = left.groupby(groupby_columns).agg(LeftBoundary = ('LeftBoundary', 'max'),
                                             LeftBound = ('Contribution', 'sum'),
                                             Minimum = ('Contribution', 'min'))
    
    # Replace the numeric value for the boundary
    left['LeftBoundary'] = left['LeftBoundary'].replace([0,1], ['Closed','Open'])
    
    # Means with infinite values produce nan values rather than np.inf values
    # Convert nan to inf only if infinite values are included in the mean
    left['LeftBound'] = np.where(left['Minimum'] == -np.inf, -np.inf, left['LeftBound'])
    
    # Determine right bound and boundary for each group
    
    # Change notation of 'Closed' and 'Open' boundaries to 1 and 0, respectively
    # Use 1 for closed to ensure that closed boundaries are sorted larger than open
    # boundaries when the right bound is tied
    df['RightBoundary'] = df['RightBoundary'].replace(['Closed','Open'], [1,0])
    
    # Sort right bound values
    right = df.copy()[groupby_columns+['Size','RightBoundary','RightBound']].sort_values(by=['RightBound','RightBoundary'])
    
    # Add index for each group
    right['Index'] = right.groupby(groupby_columns).cumcount() + 1
    
    # Determine the rank in each group for the percentile
    right['Rank'] = round(C + percentile*(right['Size'] + 1 - 2*C),8)
    
    # Generate proximity of each result to percentile rank using the index
    right['Proximity'] = 0
    
    conditions = [
        # If the percentile rank is a whole number, then use that index result
        (right['Rank'] == right['Index']),
        # If the percentile rank is less than 1 above the index value,
        # then assign the appropriate contribution to that index value
        (right['Rank'] - right['Index']).between(0,1,inclusive='neither'),
        # If the percentile rank is less than 1 below the index value,
        # then assign the appropriate contribution to that index value
        (right['Index'] - right['Rank']).between(0,1,inclusive='neither')
        ]
    
    results = [
        1,
        1 - (right['Rank'] - right['Index']),
        1 - (right['Index'] - right['Rank']),
        ]
    
    right['Proximity'] = np.select(conditions, results, np.nan)
    
    # Drop non-contributing rows
    right = right[right['Proximity'] > 0]
    
    # Calculate contribution for index values that contribute to the result
    right['Contribution'] = right['Proximity'] * right['RightBound']
    
    # Determine right bound and boundary using the sum of the contributions
    # and an open boundary if any of the contributing values is open
    right = right.groupby(groupby_columns).agg(RightBound = ('Contribution', 'sum'),
                                               Maximum = ('Contribution', 'max'),
                                               RightBoundary = ('RightBoundary', 'min'))
    
    # Replace the numeric value for the boundary
    right['RightBoundary'] = right['RightBoundary'].replace([1,0], ['Closed','Open'])
    
    # Means with infinite values produce nan values rather than np.inf values
    # Convert nan to inf only if infinite values are included in the mean
    right['RightBound'] = np.where(right['Maximum'] == np.inf, np.inf, right['RightBound'])
    
    # Merge the two boundaries to create the interval for the percentile
    # Check that the merge is 1-to-1
    df = left.merge(right, how = 'outer', on = groupby_columns, validate = '1:1')
    
    return df

def percentile(df,
               percentile,
               method = 'hazen',
               groupby_columns = [[]],
               values = ['CensorComponent','NumericComponent'],
               focus_high_potential = True,
               include_negative_interval = False,
               precision_tolerance_to_drop_censor = 0.25,
               precision_rounding = True):
    '''
    A function that combines the relevant percentile and utility functions to
    generate the percenitle results for groups within a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains censored or uncensored results
    percentile : float
        The desired percentile. Values should be between 0 and 100.
    method : string
        The percentile method. The options and definitions come from:
            https://environment.govt.nz/assets/Publications/Files/hazen-percentile-calculator-2.xls
            Options include the following, ordered from largest to smallest result.
            - weiball
            - tukey
            - blom
            - hazen
            - excel
        The default is hazen.
    groupby_columns : list of lists of strings, optional
        List of column names that should be used to create groups. A percentile
        will be found within each group. Multiple lists can be supplied to
        perform sequential median percentiles before calculating a percentile
        over the final grouping.
        The default is [[]].
    values : list of strings, optional
        The column name(s) for the column(s) that contain the results. If a 
        single column name is given, it is assumed that the column contains
        combined censor and numeric components. If two column names are
        provided, then the first should only contain one of five censors (<,≤,≥,>)
        and the second should contain only numeric data.
        The default is ['CensorComponent','NumericComponent'].
    focus_high_potential : boolean, optional
        If True, then information on the highest potential result will be
        focused over the lowest potential result.
    include_negative_interval : boolean, optional
        If True, then all positive and negative values are considered
        e.g., <0.5 would be converted to (-np.inf,5).
        If False, then only non-negative values are considered
        e.g., <0.5 would be converted to [0,5).
        This setting only affects results if focus_high_potential is False.
        The default is False.
    precision_tolerance_to_drop_censor : float, optional
        Threshold for reporting censored vs non-censored results.
        Using the default, a result that is known to be in the interval (0.3, 0.5)
        would be returned as 0.4, whereas a tolerance of 0 would yield a
        result of <0.5 or >0.3 depending on the value of focus_highest_potential.
        The default is 0.25.
    precision_rounding : boolean, optional
        If True, a rounding method is applied to round results to have no more
        decimals than what can be measured.
        The default is True.

    Returns
    -------
    df : DataFrame
        DataFrame that contains calculated percentiles

    '''
    
    # If single result column provided, then split column
    if len(values) == 1:
        censor_column = 'CensorComponent'
        numeric_column = 'NumericComponent'
        df = result_to_components(df,values[0])
    # Else define the names to use for the censor and numeric columns
    else:
        censor_column = values[0]
        numeric_column = values[1]
    
    # Convert the results from censor and numeric components to an interval representation
    df = components_to_interval(df, censor_column, numeric_column, include_negative_interval)
    
    # Cycle through each grouping
    for grouping in groupby_columns:
        # Only apply percentile to final grouping
        if grouping == groupby_columns[-1]:
            # Using the intervals, determine the range of possible percentiles
            df = percentile_interval(df, percentile, method, grouping)
        # Apply median to all prior groupings
        else:
            # Using the intervals, determine the range of possible medians
            df = percentile_interval(df, 50, method, grouping)
    
    # Create interval notation for the average
    df = interval_notation(df, precision_rounding)
    
    # Convert the interval for the minimum into censor and numeric notation
    df = interval_to_components(df, censor_column, numeric_column,
                                focus_high_potential = focus_high_potential,
                                include_negative_interval = include_negative_interval,
                                precision_tolerance_to_drop_censor = precision_tolerance_to_drop_censor)
    
    # Combine the censor and numeric components into a result
    df = components_to_result(df, censor_column, numeric_column, precision_rounding)
    
    return df

