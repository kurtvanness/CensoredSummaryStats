
import numpy as np
from .utils import string_precision, numeric_precision

# Set default column names
stat_col = 'Statistic'
result_col = 'Result'
censor_col = 'CensorComponent'
numeric_col = 'NumericComponent'
interval_col = 'Interval'
left_boundary_col = '__LeftBoundary__'
left_bound_col = '__LeftBound__'
midpoint_col = '__MidPoint__'
right_bound_col = '__RightBound__'
right_boundary_col = '__RightBoundary__'



#%% Conversion between result and interval components

#%%% Between result and censor/numeric components

def result_to_components(df,
                         result_col,
                         censor_col=censor_col,
                         numeric_col=numeric_col):
    '''
    A function that separates censors (<,≤,≥,>) from a result. Splitting the
    censor component from the numeric component enables the use of numeric
    data types. A result of <10 (string/text) is split into a censor component
    of < (string/text) and a numeric component of 10 (float/decimal).

    Parameters
    ----------
    df : DataFrame
        DataFrame containing a column to be split into components
    result_col : string
        Column name for the column that contains the results to be split
    censor_col : string
        Column name to give the created censor column.
        The default is CensorComponent.
    numeric_col : string
        Column name to give the created numeric column.
        The default is NumericComponent.

    Returns
    -------
    df : DataFrame
        Input DataFrame with two additional columns created for the censor
        and numeric components

    '''
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Ensure the value column is text/string data type.
    df[result_col] = df[result_col].astype(str)
    
    # Create censor and numeric columns from result column
    df[censor_col] = df[result_col].str.extract(r'([<≤≥>])').fillna('')
    df[numeric_col] = df[result_col].str.replace('<|≤|≥|>','',regex=True).astype(float)
    
    return df

def components_to_result(df,
                         precision_rounding = True):
    '''
    A function that combines censor and numeric components into a combined
    string/text result.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing censor and numeric columns that will be combined
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
        df[result_col] = df[censor_col] + df[numeric_col].apply(string_precision,thousands_comma=True)
        df[numeric_col] = df[numeric_col].apply(numeric_precision)
    else:
        df[result_col] = df[censor_col] + df[numeric_col].astype(str)
    
    return df

#%%% Between censor/numeric components and interval components

def components_to_interval(df,
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
    
    # Define where the left bound is closed
    df[left_boundary_col] = np.where(df[censor_col].isin(['','≥']),
                                  'Closed', 'Open')
    
    # Define where left bound is unlimited
    df[left_bound_col] = np.where(df[censor_col].isin(['<','≤']),
                               -np.inf, df[numeric_col])
    
    # Define where right bound is unlimited
    df[right_bound_col] = np.where(df[censor_col].isin(['≥','>']),
                                np.inf, df[numeric_col])
    
    # Define where the right bound is closed
    df[right_boundary_col] = np.where(df[censor_col].isin(['≤','']),
                                   'Closed', 'Open')
    
    # If left censored are assumed positive
    if not include_negative_interval:
        # Check for any negative values (exclude -inf)
        if df[left_bound_col].between(-np.inf, 0, inclusive='neither').any():
            raise ValueError('Negative values exist in the data. Resolve negative'
                             'values or set include_negative_interval to True')
        # Set any -inf left bounds to 0 with closed boundary
        condition = (df[left_bound_col] < 0)
        df[left_boundary_col] = np.where(condition, 'Closed', df[left_boundary_col])
        df[left_bound_col] = np.where(condition, 0.0, df[left_bound_col])
    
    return df


def interval_to_components(df,
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
    df[midpoint_col] = np.where((df[left_bound_col] > -np.inf) & (df[right_bound_col] < np.inf),
                              0.5 * (df[left_bound_col] + df[right_bound_col]),
                              np.nan)
    
    
    # Define the conditions for an uncensored result
    conditions = [
        # If the bounds are equal, then the result is uncensored
        (df[left_bound_col] == df[right_bound_col]),
        # If the bounds are finite and the interval is within the precision tolerance, avoid the use of censors
        (df[right_bound_col] - df[midpoint_col]) <= df[midpoint_col] * precision_tolerance_to_drop_censor
        ]
    
    censor_results = [
        '',
        ''
        ]
    
    numeric_results =[
        df[midpoint_col],
        df[midpoint_col]
        ]
    
    # If focused on the highest potential result
    if focus_high_potential:
        # Set censor and numeric components for each condition
        conditions += [
            # If there is an infinite right bound, the result is right censored
            (df[right_bound_col] == np.inf) & (df[left_bound_col] > -np.inf) & (df[left_boundary_col] == 'Open'),
            (df[right_bound_col] == np.inf) & (df[left_bound_col] > -np.inf) & (df[left_boundary_col] == 'Closed'),
            # Otherwise, the result should be left censored
            (df[right_bound_col] < np.inf) & (df[right_boundary_col] == 'Open'),
            (df[right_bound_col] < np.inf) & (df[right_boundary_col] == 'Closed')
            ]
        
        censor_results += [
            '>',
            '≥',
            '<',
            '≤'
            ]
        
        numeric_results += [
            df[left_bound_col],
            df[left_bound_col],
            df[right_bound_col],
            df[right_bound_col]
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
            (df[left_bound_col] == lower_bound) & (df[right_bound_col] < np.inf) & (df[right_boundary_col] == 'Open'),
            (df[left_bound_col] == lower_bound) & (df[right_bound_col] < np.inf) & (df[right_boundary_col] == 'Closed'),
            # Otherwise, the result should be right censored
            (df[left_bound_col] > lower_bound) & (df[left_boundary_col] == 'Open'),
            (df[left_bound_col] > lower_bound) & (df[left_boundary_col] == 'Closed')
            ]
        
        censor_results += [
            '<',
            '≤',
            '>',
            '≥'
            ]
        
        numeric_results += [
            df[right_bound_col],
            df[right_bound_col],
            df[left_bound_col],
            df[left_bound_col]
            ]
    
    # Determine the censor and numeric components
    # If no condition is met, default to <> and NaN
    df[censor_col] = np.select(conditions, censor_results, '<>')
    df[numeric_col] = np.select(conditions, numeric_results, np.nan)
    
    return df

#%%% Interval components to interval string

def interval_notation(df,
                      precision_rounding=True):
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
    df[interval_col] = np.where(df[left_boundary_col] == 'Open', '(', '[')
    
    # Incorporate left and right bounds
    if precision_rounding:
        df[interval_col] += df[left_bound_col].apply(string_precision) + ', ' + df[right_bound_col].apply(string_precision)
    else:
        df[interval_col] += df[left_bound_col].astype(str) + ', ' + df[right_bound_col].astype(str)
    
    # Determine the right boundary symbol
    df[interval_col] += np.where(df[right_boundary_col].isin(['Open']), ')', ']')
    
    return df


#%% Maximum Result

def maximum_interval(df,
                     groupby_cols):
    '''
    A function that analyses the interval notation form of results returned
    from components_to_interval to generate a new interval for the maximum. Groups
    of results can be defined by including the columns that should be used to
    create groups.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains results in a specific interval notation.
    groupby_cols : list of strings
        List of column names that should be used to create groups.

    Returns
    -------
    df : DataFrame
        DataFrame that has the interval for the maximum (for each group if
        column names are provided for grouping)

    '''
    
    # Create a copy of the DataFrame
    df = df.copy().reset_index()
    
    # Create column that indicates the generated statistic and append to grouping list
    df[stat_col] = 'Maximum'
    groupby_cols.append(stat_col)
    
    # Determine left bound and boundary for maximum for each group.
    
    # Consider the maximum bound for each left boundary option to determine
    # whether the bound should be open or closed.
    left = df.groupby(groupby_cols+[left_boundary_col])[left_bound_col].max().unstack(left_boundary_col)
    # Create missing columns
    for item in ['Open','Closed']:
        if item not in left.columns:
            left[item] = np.nan
    # Use the maximum closed boundary if there are no open boundaries or
    # if the closed boundary is larger than the largest open boundary
    condition = (left['Closed'] > left['Open']) | (left['Open'].isna())
    left[left_boundary_col] = np.where(condition,'Closed','Open')
    left[left_bound_col] = np.where(condition,left['Closed'],left['Open'])
    left = left[[left_boundary_col,left_bound_col]]
    
    # Determine right bound and boundary for maximum for each group.
    
    # Consider the maximum bound for each right boundary option to determine
    # whether the bound should be open or closed.
    right = df.groupby(groupby_cols+[right_boundary_col])[right_bound_col].max().unstack(right_boundary_col)
    # Create missing columns
    for item in ['Open','Closed']:
        if item not in right.columns:
            right[item] = np.nan
    # Use the maximum closed boundary if there are no open boundaries or
    # if the closed boundary is larger than or equal to the largest open boundary
    condition = (right['Closed'] >= right['Open']) | (right['Open'].isna())
    right[right_bound_col] = np.where(condition,right['Closed'],right['Open'])
    right[right_boundary_col] = np.where(condition,'Closed','Open')
    right = right[[right_bound_col,right_boundary_col]]
    
    # Merge the two boundaries to create the interval for the maximum
    # Check that the merge is 1-to-1
    df = left.merge(right, how = 'outer', on = groupby_cols, validate = '1:1')
    
    # Reset index
    df = df.reset_index()
    
    return df

def maximum(df,
            result_col,
            groupby_cols = None,
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
    result_col : string
        The column name for the column that contain the results as text.
        Only four possible censors should be used (<,≤,≥,>).
    groupby_cols : lists of strings (or list of lists), optional
        List of column names that should be used to create groups. A maximum
        will be found within each group. Multiple lists can be supplied to
        perform sequential maxima before converting intervals to a result.
        The default is None.
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
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Split the result into censor and numeric components
    df = result_to_components(df,result_col)
    
    # Convert the results from censor and numeric components to an interval representation
    df = components_to_interval(df, include_negative_interval)
    
    # Using the intervals, determine the range of possible maxima
    
    # If there are no groupby-columns, then take max of all results
    if not groupby_cols:
        df = maximum_interval(df, groupby_cols=[])
    # If the groupby_cols is a list of lists, then perform multiple maximums
    elif isinstance(groupby_cols[0],list):
        for grouping in groupby_cols:
            # Using the intervals, determine the range of possible maxima
            df = maximum_interval(df, grouping)
    # Else the groupby_cols is a list of column names
    else:
        df = maximum_interval(df, groupby_cols)
    
    # Convert the interval for the maximum into censor and numeric notation
    df = interval_to_components(df,
                                focus_high_potential = True,
                                include_negative_interval = include_negative_interval,
                                precision_tolerance_to_drop_censor = precision_tolerance_to_drop_censor)
    
    # Create interval notation for the maximum
    df = interval_notation(df, precision_rounding)
    
    # Combine the censor and numeric components into a result
    df = components_to_result(df, precision_rounding)
    
    # Drop working columns
    df = df.drop(df.filter(regex='__').columns, axis=1)
    
    # Reorder columns
    for col in [stat_col,result_col,censor_col,numeric_col,interval_col]:
        df[col] = df.pop(col)
    
    return df

#%% Minimum Result

def minimum_interval(df,
                     groupby_cols):
    '''
    A function that analyses the interval notation form of results returned
    from components_to_interval to generate a new interval for the minimum. Groups
    of results can be defined by including the columns that should be used to
    create groups.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains results in a specific interval notation.
    groupby_cols : list of strings
        List of column names that should be used to create groups.

    Returns
    -------
    df : DataFrame
        DataFrame that has the interval for the minimum (for each group if
        column names are provided for grouping)

    '''
    
    # Create a copy of the DataFrame
    df = df.copy().reset_index()
    
    # Create column that indicates the generated statistic and append to grouping list
    df[stat_col] = 'Minimum'
    groupby_cols.append(stat_col)
    
    # Determine left bound and boundary for minimum for each group.
    
    # Consider the minimum bound for each left boundary option to determine
    # whether the bound should be open or closed.
    left = df.groupby(groupby_cols+[left_boundary_col])[left_bound_col].min().unstack(left_boundary_col)
    # Create missing columns
    for item in ['Open','Closed']:
        if item not in left.columns:
            left[item] = np.nan
    # Use the minimum closed boundary if there are no open boundaries or
    # if the closed boundary is less than or equal to the least open boundary
    condition = (left['Closed'] <= left['Open']) | (left['Open'].isna())
    left[left_boundary_col] = np.where(condition,'Closed','Open')
    left[left_bound_col] = np.where(condition,left['Closed'],left['Open'])
    left = left[[left_boundary_col,left_bound_col]]
    
    # Determine right bound and boundary for minimum for each group.
    
    # Consider the minimum bound for each right boundary option to determine
    # whether the bound should be open or closed.
    right = df.groupby(groupby_cols+[right_boundary_col])[right_bound_col].min().unstack(right_boundary_col)
    # Create missing columns
    for item in ['Open','Closed']:
        if item not in right.columns:
            right[item] = np.nan
    # Use the minimum closed boundary if there are no open boundaries or
    # if the closed boundary is less than the least open boundary
    condition = (right['Closed'] < right['Open']) | (right['Open'].isna())
    right[right_bound_col] = np.where(condition,right['Closed'],right['Open'])
    right[right_boundary_col] = np.where(condition,'Closed','Open')
    right = right[[right_bound_col,right_boundary_col]]
    
    # Merge the two boundaries to create the interval for the minimum
    # Check that the merge is 1-to-1
    df = left.merge(right, how = 'outer', on = groupby_cols, validate = '1:1')
    
    # Reset index
    df = df.reset_index()
    
    return df

def minimum(df,
            result_col,
            groupby_cols = None,
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
    result_col : string
        The column name for the column that contain the results as text.
        Only four possible censors should be used (<,≤,≥,>).
    groupby_cols : lists of strings (or list of lists), optional
        List of column names that should be used to create groups. A minimum
        will be found within each group. Multiple lists can be supplied to
        perform sequential minima before converting intervals to a result.
        The default is None.
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
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Split the result into censor and numeric components
    df = result_to_components(df,result_col)
    
    # Convert the results from censor and numeric components to an interval representation
    df = components_to_interval(df, include_negative_interval)
    
    # Using the intervals, determine the range of possible minima
    
    # If there are no groupby-columns, then take min of all results
    if not groupby_cols:
        df = minimum_interval(df, groupby_cols=[])
    # If the groupby_cols is a list of lists, then perform multiple minimums
    elif isinstance(groupby_cols[0],list):
        for grouping in groupby_cols:
            # Using the intervals, determine the range of possible maxima
            df = minimum_interval(df, grouping)
    # Else the groupby_cols is a list of column names
    else:
        df = minimum_interval(df, groupby_cols)
    
    # Convert the interval for the minimum into censor and numeric notation
    df = interval_to_components(df,
                                focus_high_potential = False,
                                include_negative_interval = include_negative_interval,
                                precision_tolerance_to_drop_censor = precision_tolerance_to_drop_censor)
    
    # Create interval notation for the minimum
    df = interval_notation(df, precision_rounding)
    
    # Combine the censor and numeric components into a result
    df = components_to_result(df, precision_rounding)
    
    # Drop working columns
    df = df.drop(df.filter(regex='__').columns, axis=1)
    
    # Reorder columns
    for col in [stat_col,result_col,censor_col,numeric_col,interval_col]:
        df[col] = df.pop(col)
    
    return df

#%% Average Result

def average_interval(df,
                     groupby_cols):
    '''
    A function that analyses the interval notation form of results returned
    from components_to_interval to generate a new interval for the average. Groups
    of results can be defined by including the columns that should be used to
    create groups.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains results in a specific interval notation.
    groupby_cols : list of strings
        List of column names that should be used to create groups.

    Returns
    -------
    df : DataFrame
        DataFrame that has the interval for the average (for each group if
        column names are provided for grouping)

    '''
    
    # Create a copy of the DataFrame
    df = df.copy().reset_index()
    
    # Create column that indicates the generated statistic and append to grouping list
    df[stat_col] = 'Average'
    groupby_cols.append(stat_col)
    
    # Change notation of 'Closed' and 'Open' boundaries to 0 and 1, respectively
    # The presence of any open boundaries on one side ensure that the interval for the average is also
    # open on that side
    df[[left_boundary_col,right_boundary_col]] = df[[left_boundary_col,right_boundary_col]].replace(['Closed','Open'], [0,1])
    
    # Get the left/right bounds of the average by averaging bounds within the group
    # Determine whether any Open (now value of 1) boundaries exist. If there are
    # any open boundaries used in the average, then the resulting average will be open
    df = df.groupby(groupby_cols).agg(**{
                            left_boundary_col: (left_boundary_col, 'max'),
                            '__Minimum__': (left_bound_col,'min'),
                            left_bound_col: (left_bound_col,'mean'),
                            right_bound_col: (right_bound_col,'mean'),
                            '__Maximum__': (left_bound_col,'max'),
                            right_boundary_col: (right_boundary_col,'max')
                            })
    
    # Replace integers with text for boundaries
    df[[left_boundary_col,right_boundary_col]] = df[[left_boundary_col,right_boundary_col]].replace([0,1], ['Closed','Open'])
    
    # Means with infinite values produce nan values rather than np.inf values
    # Convert nan to inf only if infinite values are included in the mean
    df[left_bound_col] = np.where(df['__Minimum__'] == -np.inf, -np.inf, df[left_bound_col])
    df[right_bound_col] = np.where(df['__Maximum__'] == np.inf, np.inf, df[right_bound_col])
    
    # Reset index
    df = df.reset_index()
    
    return df

def average(df,
            result_col,
            groupby_cols = None,
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
    result_col : string
        The column name for the column that contain the results as text.
        Only four possible censors should be used (<,≤,≥,>).
    groupby_cols : lists of strings (or list of lists), optional
        List of column names that should be used to create groups. An average
        will be found within each group. Multiple lists can be supplied to
        perform sequential averaging before converting intervals to a result.
        The default is None.
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
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Split the result into censor and numeric components
    df = result_to_components(df,result_col)
    
    # Convert the results from censor and numeric components to an interval representation
    df = components_to_interval(df, include_negative_interval)
    
    # Using the intervals, determine the range of possible averages
    
    # If there are no groupby-columns, then take average of all results
    if not groupby_cols:
        df = average_interval(df, groupby_cols=[])
    # If the groupby_cols is a list of lists, then perform multiple averages
    elif isinstance(groupby_cols[0],list):
        for grouping in groupby_cols:
            # Using the intervals, determine the range of possible maxima
            df = average_interval(df, grouping)
    # Else the groupby_cols is a list of column names
    else:
        df = average_interval(df, groupby_cols)
    
    # Convert the interval for the average into censor and numeric notation
    df = interval_to_components(df,
                                focus_high_potential = focus_high_potential,
                                include_negative_interval = include_negative_interval,
                                precision_tolerance_to_drop_censor = precision_tolerance_to_drop_censor)
    
    # Create interval notation for the average
    df = interval_notation(df, precision_rounding)
    
    # Combine the censor and numeric components into a result
    df = components_to_result(df, precision_rounding)
    
    # Drop working columns
    df = df.drop(df.filter(regex='__').columns, axis=1)
    
    # Reorder columns
    for col in [stat_col,result_col,censor_col,numeric_col,interval_col]:
        df[col] = df.pop(col)
    
    return df

#%% Percentile Result

def percentile_interval(df,
                        groupby_cols,
                        percentile,
                        method = 'hazen',):
    '''
    A function that analyses the interval notation form of results returned
    from components_to_interval to generate a new interval for percentiles. Groups
    of results can be defined by including the columns that should be used to
    create groups.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains results in a specific interval notation.
    groupby_cols : list of strings
        List of column names that should be used to create groups.
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
    df[stat_col] = f'Percentile-{percentile}'
    groupby_cols.append(stat_col)
    
    # Convert percentile to be between 0 and 1
    percentile = percentile/100
    
    # Determine size of each group
    df['__Size__'] = df.groupby(groupby_cols).transform('size')
    
    # Set values for percentile methods
    method_dict = {'weiball':0.0, 'tukey':1/3, 'blom':3/8, 'hazen':1/2, 'excel':1.0}
    # https://en.wikipedia.org/wiki/Percentile
    C = method_dict[method]
    
    # Calculate minimum data size for percentile method to 
    # ensure rank is at least 1 and no more than len(data)
    minimum_size = round(C + (1-C)*max((1-percentile)/percentile,
                                       percentile/(1-percentile)),10)
    
    # Only consider groups that meet the minimum size requirement
    df = df[df['__Size__'] >= minimum_size]
    
    # Determine left bound and boundary for each group
    
    # Change notation of 'Closed' and 'Open' boundaries to 0 and 1, respectively
    # Use 0 for closed to ensure that closed boundaries are sorted less than open
    # boundaries when the left bound is tied
    df[left_boundary_col] = df[left_boundary_col].replace(['Closed','Open'], [0,1])
    
    # Sort left bound values
    left = df.copy()[groupby_cols+['__Size__',left_boundary_col,left_bound_col]].sort_values(by=[left_bound_col,left_boundary_col])
    
    # Add index for each group
    left['__Index__'] = left.groupby(groupby_cols).cumcount() + 1
    
    # Determine the rank in each group for the percentile
    left['__Rank__'] = round(C + percentile*(left['__Size__'] + 1 - 2*C),8)
    
    # Generate proximity of each result to percentile rank using the index
    left['__Proximity__'] = 0
    
    conditions = [
        # If the percentile rank is a whole number, then use that index result
        (left['__Rank__'] == left['__Index__']),
        # If the percentile rank is less than 1 above the index value,
        # then assign the appropriate contribution to that index value
        (left['__Rank__'] - left['__Index__']).between(0,1,inclusive='neither'),
        # If the percentile rank is less than 1 below the index value,
        # then assign the appropriate contribution to that index value
        (left['__Index__'] - left['__Rank__']).between(0,1,inclusive='neither')
        ]
    
    results = [
        1,
        1 - (left['__Rank__'] - left['__Index__']),
        1 - (left['__Index__'] - left['__Rank__']),
        ]
    
    left['__Proximity__'] = np.select(conditions, results, np.nan)
    
    # Drop non-contributing rows
    left = left[left['__Proximity__'] > 0]
    
    # Calculate contribution for index values that contribute to the result
    left['__Contribution__'] = left['__Proximity__'] * left[left_bound_col]
    
    # Determine left bound and boundary using the sum of the contributions
    # and an open boundary if any of the contributing values is open
    left = left.groupby(groupby_cols).agg(**{
                            left_boundary_col: (left_boundary_col, 'max'),
                            left_bound_col: ('__Contribution__','sum'),
                            '__Minimum__': ('__Contribution__','min')
                            })
    
    # Replace the numeric value for the boundary
    left[left_boundary_col] = left[left_boundary_col].replace([0,1], ['Closed','Open'])
    
    # Means with infinite values produce nan values rather than np.inf values
    # Convert nan to inf only if infinite values are included in the mean
    left[left_bound_col] = np.where(left['__Minimum__'] == -np.inf, -np.inf, left[left_bound_col])
    
    # Determine right bound and boundary for each group
    
    # Change notation of 'Closed' and 'Open' boundaries to 1 and 0, respectively
    # Use 1 for closed to ensure that closed boundaries are sorted larger than open
    # boundaries when the right bound is tied
    df[right_boundary_col] = df[right_boundary_col].replace(['Closed','Open'], [1,0])
    
    # Sort right bound values
    right = df.copy()[groupby_cols+['__Size__',right_boundary_col,right_bound_col]].sort_values(by=[right_bound_col,right_boundary_col])
    
    # Add index for each group
    right['__Index__'] = right.groupby(groupby_cols).cumcount() + 1
    
    # Determine the rank in each group for the percentile
    right['__Rank__'] = round(C + percentile*(right['__Size__'] + 1 - 2*C),8)
    
    # Generate proximity of each result to percentile rank using the index
    right['__Proximity__'] = 0
    
    conditions = [
        # If the percentile rank is a whole number, then use that index result
        (right['__Rank__'] == right['__Index__']),
        # If the percentile rank is less than 1 above the index value,
        # then assign the appropriate contribution to that index value
        (right['__Rank__'] - right['__Index__']).between(0,1,inclusive='neither'),
        # If the percentile rank is less than 1 below the index value,
        # then assign the appropriate contribution to that index value
        (right['__Index__'] - right['__Rank__']).between(0,1,inclusive='neither')
        ]
    
    results = [
        1,
        1 - (right['__Rank__'] - right['__Index__']),
        1 - (right['__Index__'] - right['__Rank__']),
        ]
    
    right['__Proximity__'] = np.select(conditions, results, np.nan)
    
    # Drop non-contributing rows
    right = right[right['__Proximity__'] > 0]
    
    # Calculate contribution for index values that contribute to the result
    right['__Contribution__'] = right['__Proximity__'] * right[right_bound_col]
    
    # Determine right bound and boundary using the sum of the contributions
    # and an open boundary if any of the contributing values is open
    right = right.groupby(groupby_cols).agg(**{
                            right_bound_col: ('__Contribution__','sum'),
                            right_boundary_col: (right_boundary_col, 'min'),
                            '__Maximum__': ('__Contribution__','max')
                            })
    
    # Replace the numeric value for the boundary
    right[right_boundary_col] = right[right_boundary_col].replace([1,0], ['Closed','Open'])
    
    # Means with infinite values produce nan values rather than np.inf values
    # Convert nan to inf only if infinite values are included in the mean
    right[right_bound_col] = np.where(right['__Maximum__'] == np.inf, np.inf, right[right_bound_col])
    
    # Merge the two boundaries to create the interval for the percentile
    # Check that the merge is 1-to-1
    df = left.merge(right, how = 'outer', on = groupby_cols, validate = '1:1')
    
    # Reset index
    df = df.reset_index()
    
    return df

def percentile(df,
               result_col,
               percentile,
               method = 'hazen',
               groupby_cols = None,
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
    result_col : string
        The column name for the column that contain the results as text.
        Only four possible censors should be used (<,≤,≥,>).
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
    groupby_cols : lists of strings (or list of lists), optional
        List of column names that should be used to create groups. A percentile
        will be found within each group. Multiple lists can be supplied to
        perform sequential median percentiles before calculating a percentile
        over the final grouping.
        The default is None.
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
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Split the result into censor and numeric components
    df = result_to_components(df,result_col)
    
    # Convert the results from censor and numeric components to an interval representation
    df = components_to_interval(df, include_negative_interval)
    
    # Using the intervals, determine the range of possible percentiles
    
    # If there are no groupby-columns, then take percentile of all results
    if not groupby_cols:
        df = percentile_interval(df, groupby_cols=[],
                                 percentile=percentile,
                                 method=method)
    # If the groupby_cols is a list of lists, then perform multiple percentiles
    elif isinstance(groupby_cols[0],list):
        for grouping in groupby_cols:
            # Only apply percentile to final grouping
            if grouping == groupby_cols[-1]:
                # Using the intervals, determine the range of possible percentiles
                df = percentile_interval(df, grouping, percentile, method)
            # Apply median to all prior groupings
            else:
                # Using the intervals, determine the range of possible medians
                df = percentile_interval(df, grouping, 50, method)
    # Else the groupby_cols is a list of column names
    else:
        df = percentile_interval(df, groupby_cols, percentile, method)
    
    # Convert the interval for the minimum into censor and numeric notation
    df = interval_to_components(df,
                                focus_high_potential = focus_high_potential,
                                include_negative_interval = include_negative_interval,
                                precision_tolerance_to_drop_censor = precision_tolerance_to_drop_censor)
    
    # Create interval notation for the percentile
    df = interval_notation(df, precision_rounding)
    
    # Combine the censor and numeric components into a result
    df = components_to_result(df, precision_rounding)
    
    # Drop working columns
    df = df.drop(df.filter(regex='__').columns, axis=1)
    
    # Reorder columns
    for col in [stat_col,result_col,censor_col,numeric_col,interval_col]:
        df[col] = df.pop(col)
    
    return df

def median(df,
           result_col,
           groupby_cols = None,
           focus_high_potential = True,
           include_negative_interval = False,
           precision_tolerance_to_drop_censor = 0.25,
           precision_rounding = True):
    '''
    A function that generates median results for groups within a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains censored or uncensored results
    result_col : string
        The column name for the column that contain the results as text.
        Only four possible censors should be used (<,≤,≥,>).
    groupby_cols : lists of strings (or list of lists), optional
        List of column names that should be used to create groups. A median
        will be found within each group. Multiple lists can be supplied to
        perform sequential median percentiles before calculating a median
        over the final grouping.
        The default is None.
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
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Call percentile function with percentile = 50
    df = percentile(df,
                    result_col,
                    50,
                    groupby_cols = groupby_cols,
                    focus_high_potential = focus_high_potential,
                    include_negative_interval = include_negative_interval,
                    precision_tolerance_to_drop_censor = precision_tolerance_to_drop_censor,
                    precision_rounding = precision_rounding)
    
    return df

