'''
The function in this module compares intervals to a threshold and determines
the percentage of values that exceed the threshold. Required information
includes (i) a CensoredData input object that contains the original data table
as well as column names, (ii) information on the threshold, (iii) the
specific column groupings used to combine data, and (iv) additional settings

'''

import numpy as np

def _determine_exceedances(cdf,
                           df,
                           threshold,
                           threshold_is_exceedance):
    
    # True or False are considered integer types so check for boolean too
    if (isinstance(threshold, bool) |
            (not isinstance(threshold,(int, float)))):
        raise ValueError(f'The value supplied to threshold must be numeric.'
            'Instead, the value supplied was '
            f'{repr(threshold)}.')
    
    # Check that threshold_is_exceedance is boolean
    if not isinstance(threshold_is_exceedance, bool):
        raise ValueError(f'The value provided for threshold_is_exceedance '
            'must be True or False. The value provided was '
            f'{threshold_is_exceedance}')
    
    # Determine exceedance (1=exceedance, 0=non-exceedance, nan=indeterminate)
    
    # Set conditions depending on whether the threshold is an exceedance
    if threshold_is_exceedance:
        # Set conditions for exceedances and non-exceedances
        conditions = [
            # Exceedance condition
            # The left bound is greater than or equal to the threshold
            (df[cdf.left_bound_col] >= threshold),
            # Non-exceedance conditions
            (
                # The right bound is less than the threshold
                (df[cdf.right_bound_col] < threshold) |
                # The right bound is equal to the threshold and open
                ((df[cdf.right_bound_col] == threshold) &
                 (df[cdf.right_boundary_col] == 'Open'))
            )
            ]
    # Set conditions when threshold is not an exceedance
    else:
        # Set conditions for exceedances and non-exceedances
        conditions = [
            # Exceedances conditions
            (
                # The left bound is greater than the threshold
                (df[cdf.left_bound_col] > threshold) |
                # If the left bound is equal to the threshold and open
                ((df[cdf.left_bound_col] == threshold) &
                 (df[cdf.left_boundary_col] == 'Open'))
            ),
            # Non-exceedance condition
            # The right bound is less than or equal to the threshold
            (df[cdf.right_bound_col] <= threshold)
            ]
    # Exceedances are indicated by integers
    # (1=exceedance, 0=non-exceedance)
    results = [
        1,
        0
        ]
    # Create column with exceedance results
    df[cdf.exceedances_col] = np.select(conditions, results, np.nan)
    
    return df
    
def _group_exceedances(cdf,
                       df,
                       groupby_cols):
    
    # Take maximum within each group
    df = (df.groupby(groupby_cols)[cdf.exceedances_col]
              .max()
              .reset_index())
    
    return df

def _percent_exceedances(cdf,
                         df,
                         threshold,
                         threshold_is_exceedance,
                         groupby_cols,
                         round_to):
    
    # Within groups, count exceedances, determinate values, and all values
    # including indeterminate
    df = (df.groupby(groupby_cols)
          .agg(**{
              cdf.exceedances_col: (cdf.exceedances_col,'sum'),
              cdf._determined_col: (cdf.exceedances_col,'count'),
              cdf._totalcount_col: (cdf.exceedances_col,'size')
            })
        )
    
    # Determine counts of non-exceedances and ignored/dropped values
    df[cdf.non_exceedances_col] = (
        df[cdf._determined_col] - df[cdf.exceedances_col]
        )
    df[cdf.ignored_col] = df[cdf._totalcount_col] - df[cdf._determined_col]
    
    # Calculate percentage and round
    df[cdf.numeric_col] = (
        round(df[cdf.exceedances_col] / df[cdf._determined_col] * 100,
              round_to)
        )
    # Create result and included count of dropped values
    df[cdf.result_col] = df[cdf.numeric_col].astype(str) + '%'
    df.loc[df[cdf.ignored_col] > 0, cdf.result_col] = (
        df[cdf.result_col] +
        ' (' + df[cdf.ignored_col].astype(str) + ' value(s) ignored)'
        )
    
    # Include threshold in output
    df[cdf.threshold_col] = f'{threshold}'
    if threshold_is_exceedance:
        df[cdf.threshold_col] = '<' + df[cdf.threshold_col]
    
    # Reset index
    df = df.reset_index()
    
    # Drop working columns
    df = df.drop([cdf._determined_col, cdf._totalcount_col], axis=1)
    
    return df

    
