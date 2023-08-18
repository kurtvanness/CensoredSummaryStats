'''
The functions in this module support the ability to calculate stats for
censored data. These functions require (i) a CensoredData input object that
contains column names to be used for the analysis, (ii) a DataFrame that
has been validated and preprocessed, (iii) the specific column grouping to
use to combine data, and (iv) additional stat specific inputs which currently
only includes percentile and method for the percentile function.

These functions all combine intervals within a group to a single interval
that represents the potential range of the desired stat based on all possible
values of the original intervals.

Functions include:
    -maximum
    -minimum
    -mean/average
    -addition
    -percentile/median

'''

import numpy as np


#%% Extrema stats (Maximum/Minimum)

def _extrema_interval(cdf, data, groupby_cols, stat_name, stat='max'):
    
    # Create a copy of the data
    df = data.copy()
    
    # Create stat column
    df[cdf.stat_col] = stat_name
    
    # Create empty list if no groups
    if groupby_cols == None:
        groupby_cols = []
    
    # Determine the bounds and boundaries for the maximum (or minimum)
    
    # Determine whether the left bound is open or closed by comparing the
    # largest (or least) open bound with the largest (or least) closed bound
    # Repeat for the right bound
    left = df.groupby(groupby_cols + [cdf.stat_col, cdf.left_boundary_col])
    right = df.groupby(groupby_cols + [cdf.stat_col, cdf.right_boundary_col])
    
    # Only keep the maximum (or minimum) for each group
    if stat == 'max':
        left = left[cdf.left_bound_col].max()
        right = right[cdf.right_bound_col].max()
    elif stat == 'min':
        left = left[cdf.left_bound_col].min()
        right = right[cdf.right_bound_col].min()
    else:
        raise ValueError('Wrong input for "stat" in _extrema_interval()')
    
    # Unstack the boundaries into their own columns
    left = left.unstack(cdf.left_boundary_col)
    right = right.unstack(cdf.right_boundary_col)
    
    # Ensure both boundary column exists
    for frame in [left, right]:
        for boundary in ['Open', 'Closed']:
            if boundary not in frame.columns:
                frame[boundary] = np.nan
    
    # Determine conditions where the boundary is Closed
    # For maximum stat,
    if stat == 'max':
        # For left bound, use maximum closed bound if it is larger than
        # the largest Open boundary.
        left_condition = (left['Closed'] > left['Open'])
        # For right bound, use maximum closed bound if it is larger than
        # or equal to the largest Open boundary.
        right_condition = (right['Closed'] >= right['Open'])
    # For minimum stat
    else:
        # For left bound, use minimum closed bound if it is less than
        # or equal to least Open boundary.
        left_condition = (left['Closed'] <= left['Open'])
        # For right bound, use minimum closed bound if it is less than
        # the least Open boundary.
        right_condition = (right['Closed'] < right['Open'])
    
    # Always Closed boundary if there is no Open boundary
    left_condition |= left['Open'].isna()
    right_condition |= right['Open'].isna()
    
    # Set left bounds and boundaries based on the conditions
    left[cdf.left_boundary_col] = (
        np.where(left_condition,'Closed','Open'))
    
    left[cdf.left_bound_col] = (
        np.where(left_condition,left['Closed'],left['Open']))
    
    left = left[[cdf.left_boundary_col, cdf.left_bound_col]]
    
    # Set right bounds and boundaries based on the conditions
    right[cdf.right_bound_col] = (
        np.where(right_condition,right['Closed'],right['Open']))
    
    right[cdf.right_boundary_col] = (
        np.where(right_condition,'Closed','Open'))
    
    right = right[[cdf.right_bound_col, cdf.right_boundary_col]]
    
    # Merge the two dataframes to create the full interval
    # Check the merge is 1-to-1
    df = left.merge(right,
                    how = 'outer',
                    on = groupby_cols + [cdf.stat_col],
                    validate = '1:1')
    
    # Reset index
    df = df.reset_index()
    
    return df
    
def _maximum_interval(cdf, data, groupby_cols, stat_name='Maximum'):
    
    return _extrema_interval(cdf, data, groupby_cols, stat_name, stat='max')

def _minimum_interval(cdf, data, groupby_cols, stat_name='Minimum'):
    
    return _extrema_interval(cdf, data, groupby_cols, stat_name, stat='min')

#%% Addition based stats (Mean/Average/Sum)

def _mean_or_sum_interval(cdf, data, groupby_cols, stat_name, stat='mean'):
    
    # Create a copy of the data
    df = data.copy()
    
    # Create stat column
    df[cdf.stat_col] = stat_name
    
    # Check stat name
    if stat not in ['mean', 'sum']:
        raise ValueError('Wrong input for "stat" in _mean_or_sum_interval()')
    
    # Create empty list if no groups
    if groupby_cols == None:
        groupby_cols = []
    
    # Change notation of 'Closed' and 'Open' boundaries to 0 and 1, 
    # respectively. The presence of any open boundaries on one side ensures
    # that the interval for the result is also open on that side
    df[[cdf.left_boundary_col, cdf.right_boundary_col]] = (
        df[[cdf.left_boundary_col, cdf.right_boundary_col]]
            .replace(['Closed','Open'], [0,1]))
    
    # Get the left/right bounds of the result by averaging or adding all 
    # left/right bounds within the group.
    # Determine whether any Open (now value of 1) boundaries exist by
    # taking max of boundary values. If there are any open boundaries used in 
    # the bound mean/sum, then the resulting mean/sum should be open
    # Determine the min and max of the bounds to correct situations where
    # the bound is infinite
    df = (
        df.groupby(groupby_cols + [cdf.stat_col])
            .agg(**{
                cdf.left_boundary_col: (cdf.left_boundary_col, 'max'),
                cdf._minimum_col: (cdf.left_bound_col, 'min'),
                cdf.left_bound_col: (cdf.left_bound_col, stat),
                cdf.right_bound_col: (cdf.right_bound_col, stat),
                cdf._maximum_col: (cdf.right_bound_col, 'max'),
                cdf.right_boundary_col: (cdf.right_boundary_col, 'max')
                })
        )
    
    # Replace integers with text for boundaries
    df[[cdf.left_boundary_col, cdf.right_boundary_col]] = (
        df[[cdf.left_boundary_col, cdf.right_boundary_col]]
            .replace([0,1], ['Closed','Open']))
    
    # Means/sums with infinite values produce nan values rather than 
    # np.inf values. Convert nan to inf only if infinite values are
    # included in the mean/sum
    df[cdf.left_bound_col] = (
        np.where(df[cdf._minimum_col] == -np.inf,
                 -np.inf,
                 df[cdf.left_bound_col]
                 )
        )
    df[cdf.right_bound_col] = (
        np.where(df[cdf._maximum_col] == np.inf,
                 np.inf,
                 df[cdf.right_bound_col]
                 )
        )
    
    # Drop working columns
    df = df.drop([cdf._minimum_col, cdf._maximum_col], axis=1)
    
    # Reset index
    df = df.reset_index()
    
    return df

def _mean_interval(cdf, data, groupby_cols, stat_name='Mean'):
    
    return _mean_or_sum_interval(cdf, data, groupby_cols,
                                 stat_name, stat='mean')

def _sum_interval(cdf, data, groupby_cols, stat_name='Sum'):
    
    return _mean_or_sum_interval(cdf, data, groupby_cols,
                                 stat_name, stat='sum')

#%% Percentile stats (Percentile/Median)

def _percentile_interval(cdf,
                         data,
                         percentile,
                         groupby_cols,
                         stat_name = 'Percentile',
                         method = 'hazen'):
    
    # Create a copy of the data
    df = data.copy()
    
    # Create stat column
    df[cdf.stat_col] = stat_name
    
    # Create empty list if no groups
    if groupby_cols == None:
        groupby_cols = []
    
    # Set values for percentile methods
    method_dict = {'weiball': 0.0,
                   'tukey': 1/3,
                   'blom': 3/8,
                   'hazen': 1/2,
                   'excel':1.0}
    # https://en.wikipedia.org/wiki/Percentile
    C = method_dict[method]
    
    # Convert percentile to be between 0 and 1
    percentile = percentile/100
    
    # Create column for the size of each group
    df[cdf._size_col] = (df.groupby(groupby_cols + [cdf.stat_col])
                        .transform('size'))
    
    # Determine the rank in each group for the percentile
    # Use rounding at 8 decimals to prevent artificial decimals
    df[cdf._rank_col] = round(C + percentile*(df[cdf._size_col] + 1 - 2*C), 8)
    
    # Create warning column to flag if the percentile rank is outside
    # the range of possible values. If rank is less than 1, the minimum will
    # be returned for the percentile; if the rank is greater than the size
    # of the group, the maximum will be returned.
    df[cdf.warning_col] = ''
    max_condition = (df[cdf._rank_col] > df[cdf._size_col])
    min_condition = (df[cdf._rank_col] < 1)
    df.loc[max_condition, cdf.warning_col] = '(low count, used max)'
    df.loc[min_condition, cdf.warning_col] = '(low count, used min)'
    df.loc[max_condition, cdf._rank_col] = df[cdf._size_col]
    df.loc[min_condition, cdf._rank_col] = 1
    
    # For the left bound, change notation of Closed and Open boundaries to 0
    # and 1, respectively. Use 0 for Closed to ensure that Closed boundaries
    # are sorted smaller than Open boundaries when the left bound is tied
    df[cdf.left_boundary_col] = (
        df[cdf.left_boundary_col].replace(['Closed','Open'], [0,1]))
    
    # For the right bound, change notation of Closed and Open boundaries to 1
    # and 0, respectively. Use 1 for Closed to ensure that Closed boundaries
    # are sorted larger than Open boundaries when the right bound is tied
    df[cdf.right_boundary_col] = (
        df[cdf.right_boundary_col].replace(['Closed','Open'], [1,0]))
    
    # Sort the left and right bounds
    # Reduce working dataframes to relevant columns
    left = (
        df.copy()[groupby_cols + [cdf.stat_col] +
                  [cdf._size_col, cdf._rank_col, cdf.warning_col] +
                  [cdf.left_boundary_col, cdf.left_bound_col]]
            .sort_values(
                by = [cdf.left_bound_col, cdf.left_boundary_col]
                )
        )
    right = (
        df.copy()[groupby_cols + [cdf.stat_col] +
                  [cdf._size_col, cdf._rank_col, cdf.warning_col] +
                  [cdf.right_boundary_col, cdf.right_bound_col]]
            .sort_values(
                by = [cdf.right_bound_col, cdf.right_boundary_col]
                )
        )
    
    # Add index for each group
    left[cdf._index_col] = (left.groupby(groupby_cols + [cdf.stat_col])
                             .cumcount() + 1)
    right[cdf._index_col] = (right.groupby(groupby_cols + [cdf.stat_col])
                               .cumcount() + 1)
    
    def find_values_within_one_of_rank(df):
        
        # Determine proximity of each result to percentile rank using the index
        
        # Set default as 0
        df[cdf._proximity_col] = 0
        
        condition_set = [
            # If the percentile rank is a whole number,
            # then use that index result
            (df[cdf._rank_col] == df[cdf._index_col]),
            # If the percentile rank is less than 1 above the index value,
            # then assign the appropriate contribution to that index value
            ((df[cdf._rank_col] - df[cdf._index_col])
                                         .between(0,1,inclusive='neither')),
            # If the percentile rank is less than 1 below the index value,
            # then assign the appropriate contribution to that index value
            ((df[cdf._index_col] - df[cdf._rank_col])
                                         .between(0,1,inclusive='neither'))
            ]
        
        proximities = [
            1,
            1 - (df[cdf._rank_col] - df[cdf._index_col]),
            1 - (df[cdf._index_col] - df[cdf._rank_col])
            ]
        
        # Set proximity of each result to rank
        df[cdf._proximity_col] = np.select(condition_set, proximities, np.nan)
        
        # Drop non-contributing rows
        df = df[df[cdf._proximity_col] > 0]
        
        return df
    
    left = find_values_within_one_of_rank(left)
    right = find_values_within_one_of_rank(right)
    
    # Calculate contribution for using proximity to rank
    left[cdf._contribution_col] = (
        left[cdf._proximity_col] * left[cdf.left_bound_col])
    right[cdf._contribution_col] = (
        right[cdf._proximity_col] * right[cdf.right_bound_col])
    
    # Determine bound and boundary using the sum of the contributions
    # and an open boundary if any of the contributing values is open
    # Replace nan sums with infinite bound
    left = (
        left.groupby(groupby_cols + [cdf.stat_col, cdf.warning_col])
            .agg(**{
                cdf.left_boundary_col: (cdf.left_boundary_col, 'max'),
                cdf.left_bound_col: (cdf._contribution_col, 'sum'),
                cdf._minimum_col: (cdf._contribution_col, 'min')
                }))
    right = (
        right.groupby(groupby_cols + [cdf.stat_col, cdf.warning_col])
            .agg(**{
                cdf.right_bound_col: (cdf._contribution_col, 'sum'),
                cdf.right_boundary_col: (cdf.right_boundary_col, 'min'),
                cdf._maximum_col: (cdf._contribution_col, 'max')
                }))
    
    # Replace the numeric value for the boundary
    left[cdf.left_boundary_col] = (
        left[cdf.left_boundary_col].replace([0,1], ['Closed','Open']))
    right[cdf.right_boundary_col] = (
        right[cdf.right_boundary_col].replace([1,0], ['Closed','Open']))
    
    # Replace nan bounds with infinite bound
    left[cdf.left_bound_col] = (
        np.where(left[cdf._minimum_col] == -np.inf, -np.inf,
                 left[cdf.left_bound_col]))
    right[cdf.right_bound_col] = (
        np.where(right[cdf._maximum_col] == np.inf, np.inf,
                 right[cdf.right_bound_col]))
    
    
    # Merge the two boundaries to create the interval for the percentile
    # Check that the merge is 1-to-1
    df = left.merge(right,
                    how = 'outer',
                    on = groupby_cols + [cdf.stat_col, cdf.warning_col],
                    validate = '1:1'
                    )
    
    # Drop working columns
    df = df.drop([cdf._minimum_col, cdf._maximum_col], axis=1)
    
    # Reset index
    df = df.reset_index()
    
    return df

def _median_interval(cdf, data, groupby_cols, stat_name='Median'):
    
    return _percentile_interval(cdf, data, 50, groupby_cols, stat_name)