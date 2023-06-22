
import numpy as np
import pandas as pd
from dataclasses import dataclass
import copy
from censoredsummarystats.utils import string_precision, numeric_precision

@dataclass
class CensoredData:
    data: pd.core.frame.DataFrame
    value_col: str
    include_negative_interval: bool = False
    focus_high_potential: bool = True
    precision_tolerance_to_drop_censor: float = 0.25
    precision_rounding: bool = True
    stat_col: str = 'Statistic'
    result_col: str = 'Result'
    censor_col: str = 'CensorComponent'
    numeric_col: str = 'NumericComponent'
    left_boundary_col: str = 'LeftBoundary'
    left_bound_col: str = 'LeftBound'
    right_bound_col: str = 'RightBound'
    right_boundary_col: str = 'RightBoundary'
    interval_col: str = 'Interval'
    threshold_col: str = 'Threshold'
    exceedances_col: str = 'Exceedances'
    non_exceedances_col: str = 'NonExceedances'
    ignored_col: str = 'Ignored'
    
    def __post_init__(self):
        
        # Ensure that provided data is a copy and not indexed
        self.data = self.data.copy().reset_index()
        
        #%% Check column names
        
        # Check that generated columns don't conflict w/ existing column names
        created_columns = [attribute for attribute in self.__annotations__
                                   if attribute.endswith('_col') and
                                       attribute != 'value_col']
        for col in created_columns:
            col_name = getattr(self, col)
            if (col_name in self.data.columns) & (col_name != self.value_col):
                raise ValueError('The data contains a column named '
                                 f'"{col_name}" which may be used in the '
                                 'output. Either rename this column '
                                 'or provide an alternative value for '
                                 f'{col} when initialising CensoredDataFrame.')
        
        #%% Split value into censor/numeric components
        
        # Check if value column is not string type
        if self.data[self.value_col].dtype != str:
            # If not, convert to string
            self.data[self.value_col] = self.data[self.value_col].astype(str)
        
        # Create censor column from value column
        self.data[self.censor_col] = (self.data[self.value_col]
                                      .str.extract(r'([<≤≥>])')
                                      .fillna('')
                                      )
        # Check that there is only a single censor component
        if self.data[self.censor_col].str.len().max() > 1:
            raise ValueError('A value exists in the data with multiple '
                             'censor components.')
        # Create numeric column from value column
        self.data[self.numeric_col] = (self.data[self.value_col]
                                       .str.replace('<|≤|≥|>','',regex=True)
                                       .astype(float)
                                       )
        
        #%% Create interval for value (left/right bounds)
        
        # Define where the left bound is closed
        self.data[self.left_boundary_col] = (
                np.where(self.data[self.censor_col].isin(['','≥']),
                         'Closed',
                         'Open'
                         )
                )
        
        # Define where the left bound is unlimited
        self.data[self.left_bound_col] = (
                np.where(self.data[self.censor_col].isin(['<','≤']),
                                   -np.inf,
                                   self.data[self.numeric_col]
                                   )
                )
        
        # Define where the right bound is unlimited
        self.data[self.right_bound_col] = (
                np.where(self.data[self.censor_col].isin(['≥','>']),
                                   np.inf,
                                   self.data[self.numeric_col]
                                   )
                )
        
        # Define where the right bound is closed
        self.data[self.right_boundary_col] = (
                np.where(self.data[self.censor_col].isin(['≤','']),
                         'Closed',
                         'Open'
                         )
                )
        
        # If left censored are assumed positive
        if not self.include_negative_interval:
            # Check for any negative values (exclude -inf)
            if (self.data[self.left_bound_col]
                    .between(-np.inf, 0, inclusive='neither')
                    .any()
                    ):
                raise ValueError('Negative values exist in the data. Resolve '
                                 'negative values or set '
                                 'include_negative_interval = True')
            # Set any -inf left bounds to 0 with closed boundary
            condition = (self.data[self.left_bound_col] < 0)
            self.data[self.left_boundary_col] = (
                np.where(condition,
                         'Closed',
                         self.data[self.left_boundary_col]
                         )
                )
            self.data[self.left_bound_col] = (
                np.where(condition,
                         0.0,
                         self.data[self.left_bound_col]
                         )
                )
    #%% Utility methods
    
    def interval_notation(self, inplace = False):
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Determine the left boundary symbol
        df[self.interval_col] = (
            np.where(df[self.left_boundary_col] == 'Open',
                     '(',
                     '['
                     )
            )
        
        # Incorporate left and right bounds
        if self.precision_rounding:
            df[self.interval_col] += (
                df[self.left_bound_col].apply(string_precision) + ', ' + 
                df[self.right_bound_col].apply(string_precision)
                )
        else:
            df[self.interval_col] += (
                df[self.left_bound_col].astype(str) + ', ' + 
                df[self.right_bound_col].astype(str)
                )
        
        # Determine the right boundary symbol
        df[self.interval_col] += (
            np.where(df[self.right_boundary_col] == 'Open',
                     ')',
                     ']'
                     )
            )
        
        # Replace or return result
        if inplace:
            self.data = df
        else:
            return df
    
    def components_from_interval(self,
                                 focus_high_potential,
                                 inplace = False):
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Determine the midpoint for the interval if finite interval
        df['__MidPoint__'] = (
            np.where((df[self.left_bound_col] > -np.inf) & 
                     (df[self.right_bound_col] < np.inf),
                0.5 * (df[self.left_bound_col] + df[self.right_bound_col]),
                np.nan
                )
            )
        
        # Start with the conditions that produce an uncensored result
        conditions = [
            # If the bounds are equal, then the result is uncensored
            (df[self.left_bound_col] == 
             df[self.right_bound_col]),
            # If the bounds are finite and the interval is within the 
            # precision tolerance, then the result is uncensored
            (
                (df[self.right_bound_col] - df['__MidPoint__']) <= 
                df['__MidPoint__'] * 
                                    self.precision_tolerance_to_drop_censor)
            ]
        
        censor_results = [
            '',
            ''
            ]
        
        numeric_results =[
            df['__MidPoint__'],
            df['__MidPoint__']
            ]
        
        # If focused on the highest potential result
        if focus_high_potential:
            # Set censor and numeric components for each condition
            conditions += [
                # If there is an infinite right bound,
                # then the result is right censored
                # where closed boundary indicates potential equality
                (
                    (df[self.right_bound_col] == np.inf) & 
                    (df[self.left_bound_col] > -np.inf) & 
                    (df[self.left_boundary_col] == 'Open')
                ),
                (
                    (df[self.right_bound_col] == np.inf) & 
                    (df[self.left_bound_col] > -np.inf) & 
                    (df[self.left_boundary_col] == 'Closed')
                ),
                # Otherwise, the result is left censored
                # where closed boundary indicates potential equality
                (
                    (df[self.right_bound_col] < np.inf) & 
                    (df[self.right_boundary_col] == 'Open')
                ),
                (
                    (df[self.right_bound_col] < np.inf) & 
                    (df[self.right_boundary_col] == 'Closed')
                )
                ]
            
            censor_results += [
                '>',
                '≥',
                '<',
                '≤'
                ]
            
            numeric_results += [
                df[self.left_bound_col],
                df[self.left_bound_col],
                df[self.right_bound_col],
                df[self.right_bound_col]
                ]
        # Else focused on the lowest potential result
        else:
            # Determine the lower bound
            if self.include_negative_interval:
                lower_bound = -np.inf
            else:
                lower_bound = 0.0
            # Set censor and numeric components for each condition
            conditions += [
                # If the left bound is identical to the lower bound,
                # then the result is left censored
                # where closed boundary indicates potential equality
                (
                    (df[self.left_bound_col] == lower_bound) & 
                    (df[self.right_bound_col] < np.inf) & 
                    (df[self.right_boundary_col] == 'Open')
                ),
                (
                    (df[self.left_bound_col] == lower_bound) & 
                    (df[self.right_bound_col] < np.inf) & 
                    (df[self.right_boundary_col] == 'Closed')
                ),
                # Otherwise, the result is right censored
                # where closed boundary indicates potential equality
                (
                    (df[self.left_bound_col] > lower_bound) &
                    (df[self.left_boundary_col] == 'Open')
                ),
                (
                    (df[self.left_bound_col] > lower_bound) &
                    (df[self.left_boundary_col] == 'Closed')
                )
                ]
            
            censor_results += [
                '<',
                '≤',
                '>',
                '≥'
                ]
            
            numeric_results += [
                df[self.right_bound_col],
                df[self.right_bound_col],
                df[self.left_bound_col],
                df[self.left_bound_col]
                ]
        
        # Determine the censor and numeric components
        # If no condition is met, default to <> and NaN
        df[self.censor_col] = np.select(conditions,
                                               censor_results,
                                               '<>')
        df[self.numeric_col] = np.select(conditions,
                                                numeric_results,
                                                np.nan)
        
        # Replace or return result
        if inplace:
            self.data = df
        else:
            return df
        
    def result_from_components(self, inplace = False):
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Combine the censor and numeric components to create a combined result
        # Apply the appropriate rounding functions if precision_rounding = True
        if self.precision_rounding:
            df[self.result_col] = (
                df[self.censor_col] + 
                df[self.numeric_col]
                    .apply(string_precision, thousands_comma = True)
                )
            df[self.numeric_col] = (
                df[self.numeric_col]
                    .apply(numeric_precision)
                )
        else:
            df[self.result_col] = (
                df[self.censor_col] + 
                df[self.numeric_col].astype(str)
                )
        
        # Replace or return result
        if inplace:
            self.data = df
        else:
            return df
        
    def merge_counts(self,
                     stat_data,
                     count_cols,
                     groupby_cols = None):
        
        # Check that count_cols is a list
        if not isinstance(count_cols, list):
            raise ValueError('count_cols must be a list. '
                             f'{count_cols=} was passed.')
        # Check that items are all strings
        if not (all(isinstance(name, str) for name in count_cols)):
            raise ValueError('count_cols must be a list of strings. '
                                 f'{count_cols=} was passed.')
        # Check that names are unique
        if (len(count_cols) != len(set(count_cols))):
            raise ValueError('count_cols must contain unique strings. '
                             f'{count_cols=} was passed.')
        
        # Create a copy of the data
        df = self.data.copy()
        
        # If groupby_cols is None
        if groupby_cols == None:
            # Check that single count_col provided
            if len(count_cols) != 1:
                raise ValueError('A single column name should be provided as '
                    'a list when groupby_cols is not provided as an input. '
                    f'{count_cols=} was passed.')
            else:
                # Output length of data as count in stat_data
                stat_data[count_cols[0]] = len(df)
        # If groupby_cols is list of strings,
        elif (all(isinstance(name, str) for name in groupby_cols)):
            # Check that single count_col provided
            if len(count_cols) != 1:
                raise ValueError('A single column name should be provided as '
                    'a list when groupby_cols is a single list. '
                    f'{count_cols=} was passed.')
            else:
                # Output size of groups
                df = (df.groupby(groupby_cols, as_index=False)
                              .size()
                              .rename(columns={'size': count_cols[0]}))
                # Merge count info to stat data
                stat_data = stat_data.merge(df,
                                            how='outer',
                                            on=groupby_cols)
        # Else groupby_cols is list of lists
        else:
            # Check the list lengths are identical
            if (len(count_cols) != len(groupby_cols)):
                raise ValueError('count_cols must have the same number of '
                                 'column names as there are grouping lists in '
                                 'groupby_cols.')
            else:
                # Loop through list of groupings and sum counts
                for i in range(len(groupby_cols)):
                    df[count_cols[i]] = 1
                    df = (df.groupby(groupby_cols[i])[count_cols[:i+1]]
                              .sum()
                              .reset_index())
                # Merge count info to stat data using last grouping
                stat_data = stat_data.merge(df,
                                            how='outer',
                                            on=groupby_cols[-1])
        return stat_data
        
        
    #%% Statistic interval methods
    
    def maximum_interval(self, groupby_cols, inplace = False):
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Determine left bound and boundary for maximum of each group.
        
        # Determine whether the largest left bound is open or closed by
        # comparing the largest open and largest closed bound
        left = (
            df.groupby(groupby_cols + [self.left_boundary_col])
                [self.left_bound_col]
                .max()
                .unstack(self.left_boundary_col)
            )
        # Ensure both boundary types exist
        for item in ['Open','Closed']:
            if item not in left.columns:
                left[item] = np.nan
        # Use the maximum closed boundary if there are no open boundaries or
        # if the closed boundary is larger than the largest open boundary
        condition = (left['Closed'] > left['Open']) | (left['Open'].isna())
        left[self.left_boundary_col] = (
            np.where(condition,
                     'Closed',
                     'Open'
                     )
            )
        left[self.left_bound_col] = (
            np.where(condition,
                     left['Closed'],
                     left['Open']
                     )
            )
        left = left[[self.left_boundary_col, self.left_bound_col]]
        
        # Determine right bound and boundary for maximum of each group.
        
        
        # Determine whether the largest right bound is open or closed by
        # comparing the largest open and largest closed bound
        right = (
            df.groupby(groupby_cols + [self.right_boundary_col])
                [self.right_bound_col]
                .max().
                unstack(self.right_boundary_col)
            )
        # Ensure both boundary types exist
        for item in ['Open','Closed']:
            if item not in right.columns:
                right[item] = np.nan
        # Use the maximum closed boundary if there are no open boundaries or
        # if the closed boundary is larger than or equal to the largest 
        # open boundary
        condition = (right['Closed'] >= right['Open']) | (right['Open'].isna())
        right[self.right_bound_col] = (
            np.where(condition,
                     right['Closed'],
                     right['Open']
                     )
            )
        right[self.right_boundary_col] = (
            np.where(condition,
                     'Closed',
                     'Open'
                     )
            )
        right = right[[self.right_bound_col, self.right_boundary_col]]
        
        # Merge the two boundaries to create the interval for the maximum
        # Check that the merge is 1-to-1
        df = left.merge(right,
                        how = 'outer',
                        on = groupby_cols,
                        validate = '1:1')
        
        # Create a column that indicates the generated statistic
        df[self.stat_col] = 'Maximum'
        
        # Reorder columns
        df = df[[self.stat_col,
                 self.left_boundary_col,
                 self.left_bound_col,
                 self.right_bound_col,
                 self.right_boundary_col
                 ]]
        
        # Reset index
        df = df.reset_index()
        
        # Replace or return result
        if inplace:
            self.data = df
        else:
            return df
    
    def minimum_interval(self, groupby_cols, inplace = False):
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Determine left bound and boundary for minimum of each group.
        
        # Determine whether the least left bound is open or closed by
        # comparing the least open and least closed bound
        left = (
            df.groupby(groupby_cols + [self.left_boundary_col])
                [self.left_bound_col]
                .min()
                .unstack(self.left_boundary_col)
            )
        # Ensure both boundary types exist
        for item in ['Open','Closed']:
            if item not in left.columns:
                left[item] = np.nan
        # Use the minimum closed boundary if there are no open boundaries or
        # if the closed boundary is less than or equal to the least
        # open boundary
        condition = (left['Closed'] <= left['Open']) | (left['Open'].isna())
        left[self.left_boundary_col] = (
            np.where(condition,
                     'Closed',
                     'Open'
                     )
            )
        left[self.left_bound_col] = (
            np.where(condition,
                     left['Closed'],
                     left['Open']
                     )
            )
        left = left[[self.left_boundary_col, self.left_bound_col]]
                        
        # Determine right bound and boundary for minimum of each group.
                
        # Determine whether the least right bound is open or closed by
        # comparing the least open and least closed bound
        right = (
            df.groupby(groupby_cols + [self.right_boundary_col])
                [self.right_bound_col]
                .min().
                unstack(self.right_boundary_col)
            )
        # Ensure both boundary types exist
        for item in ['Open','Closed']:
            if item not in right.columns:
                right[item] = np.nan
        # Use the minimum closed boundary if there are no open boundaries or
        # if the closed boundary is less than the least open boundary
        condition = (right['Closed'] < right['Open']) | (right['Open'].isna())
        right[self.right_bound_col] = (
            np.where(condition,
                     right['Closed'],
                     right['Open']
                     )
            )
        right[self.right_boundary_col] = (
            np.where(condition,
                     'Closed',
                     'Open'
                     )
            )
        right = right[[self.right_bound_col, self.right_boundary_col]]
        
        # Merge the two boundaries to create the interval for the maximum
        # Check that the merge is 1-to-1
        df = left.merge(right,
                        how = 'outer',
                        on = groupby_cols,
                        validate = '1:1')
        
        # Create a column that indicates the generated statistic
        df[self.stat_col] = 'Minimum'
        
        # Reorder columns
        df = df[[self.stat_col,
                 self.left_boundary_col,
                 self.left_bound_col,
                 self.right_bound_col,
                 self.right_boundary_col
                 ]]
        
        # Reset index
        df = df.reset_index()
        
        # Replace or return result
        if inplace:
            self.data = df
        else:
            return df
    
    def mean_or_sum_interval(self,
                             stat,
                             groupby_cols,
                             inplace = False):
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Check for valid stat functions
        if stat not in ['mean','sum']:
            raise ValueError('It is not recommended for users to use the '
                'function mean_or_sum_interval. Use mean_interval or '
                'sum_interval instead. To use this function the value for '
                '"stat" must be "mean" or "sum". '
                f'The provided value was "{stat}".')
        
        # Create a column that indicates the generated statistic 
        df[self.stat_col] = stat.capitalize()
        
        # Change notation of 'Closed' and 'Open' boundaries to 0 and 1, 
        # respectively. The presence of any open boundaries on one side ensures
        # that the interval for the result is also open on that side
        df[[self.left_boundary_col, self.right_boundary_col]] = (
            df[[self.left_boundary_col, self.right_boundary_col]]
                .replace(['Closed','Open'], [0,1])
            )
        
        # Get the left/right bounds of the result by averaging or adding all 
        # left/right bounds within the group.
        # Determine whether any Open (now value of 1) boundaries exist.
        # If there are any open boundaries used in the bound mean/sum,
        # then the resulting mean/sum will be open
        df = (
            df.groupby(groupby_cols + [self.stat_col])
                .agg(**{
                    self.left_boundary_col: (self.left_boundary_col, 'max'),
                    '__Minimum__': (self.left_bound_col, 'min'),
                    self.left_bound_col: (self.left_bound_col, stat),
                    self.right_bound_col: (self.right_bound_col, stat),
                    '__Maximum__': (self.right_bound_col, 'max'),
                    self.right_boundary_col: (self.right_boundary_col, 'max')
                    })
            )
        
        # Replace integers with text for boundaries
        df[[self.left_boundary_col, self.right_boundary_col]] = (
            df[[self.left_boundary_col, self.right_boundary_col]]
                .replace([0,1], ['Closed','Open'])
            )
        
        # Means/sums with infinite values produce nan values rather than 
        # np.inf values. Convert nan to inf only if infinite values are
        # included in the mean/sum
        df[self.left_bound_col] = (
            np.where(df['__Minimum__'] == -np.inf,
                     -np.inf,
                     df[self.left_bound_col]
                     )
            )
        df[self.right_bound_col] = (
            np.where(df['__Maximum__'] == np.inf,
                     np.inf,
                     df[self.right_bound_col]
                     )
            )
        
        # Reorder columns
        df = df[[self.left_boundary_col,
                 self.left_bound_col,
                 self.right_bound_col,
                 self.right_boundary_col
                 ]]
        
        # Reset index
        df = df.reset_index()
        
        # Replace or return result
        if inplace:
            self.data = df
        else:
            return df
    
    def mean_interval(self, groupby_cols, inplace = False):
        
        return self.mean_or_sum_interval('mean', groupby_cols, inplace)
    
    def add_interval(self, groupby_cols, inplace = False):
        
        return self.mean_or_sum_interval('sum', groupby_cols, inplace)
    
    def percentile_interval(self,
                            groupby_cols,
                            percentile,
                            method = 'hazen',
                            inplace = False):
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Convert percentile to be between 0 and 1
        percentile = percentile/100
        
        # Determine size of each group
        if groupby_cols == None:
            df['__Size__'] = len(df)
        else:
            df['__Size__'] = df.groupby(groupby_cols).transform('size')
        
        # Set values for percentile methods
        method_dict = {'weiball': 0.0,
                       'tukey': 1/3,
                       'blom': 3/8,
                       'hazen': 1/2,
                       'excel':1.0}
        # https://en.wikipedia.org/wiki/Percentile
        C = method_dict[method]
        
        # Calculate minimum data size for percentile method to 
        # ensure rank is at least 1 and no more than len(data)
        minimum_size = round(C + (1-C)*max((1-percentile)/percentile,
                                           percentile/(1-percentile)),10)
        
        # Only consider groups that meet the minimum size requirement
        # df = df[df['__Size__'] >= minimum_size]
        df['__Warning__'] = ''
        condition = (df['__Size__'] < minimum_size)
        if percentile > 0.5:
            df.loc[condition, '__Warning__'] = ' (low count, used max)'
        else:
            df.loc[condition, '__Warning__'] = ' (low count, used min)'
        
        # Determine left bound and boundary for each group
        
        # Change notation of 'Closed' and 'Open' boundaries to 0 and 1,
        # respectively. Use 0 for closed to ensure that closed boundaries are
        # sorted less than open boundaries when the left bound is tied
        df[self.left_boundary_col] = (
            df[self.left_boundary_col]
                .replace(['Closed','Open'], [0,1])
            )
        
        # Sort left bound values
        grouping = ['__Size__', self.left_boundary_col, self.left_bound_col]
        if groupby_cols != None:
            grouping = groupby_cols + grouping
        
        left = (
            df.copy()[grouping]
                .sort_values(
                    by = [self.left_bound_col, self.left_boundary_col]
                    )
            )
        
        # Add index for each group
        if groupby_cols == None:
            left['__Index__'] = range(1, 1+len(left))
        else:
            left['__Index__'] = left.groupby(groupby_cols).cumcount() + 1
        
        # Determine the rank in each group for the percentile
        # Use rounding at 8 decimals to prevent artificial decimals
        left['__Rank__'] = round(
            C + percentile*(left['__Size__'] + 1 - 2*C),
            8
            )
        
        # Handle situations where rank is outside the range 1 to length of data
        conditions = [
            # Use maximum rank if rank is larger than the length of the dataset
            (left['__Rank__'] > left['__Size__']),
            # Use rank of 1 if rank is less than 1
            (left['__Rank__'] < 1)
            ]
        
        results = [
            left['__Size__'],
            1
            ]
        
        left['__Rank__'] = np.select(conditions, results, left['__Rank__'])
        
        # Generate proximity of each result to percentile rank using the index
        left['__Proximity__'] = 0
        
        conditions = [
            # If the percentile rank is a whole number,
            # then use that index result
            (left['__Rank__'] == left['__Index__']),
            # If the percentile rank is less than 1 above the index value,
            # then assign the appropriate contribution to that index value
            ((left['__Rank__'] - left['__Index__'])
                                         .between(0,1,inclusive='neither')),
            # If the percentile rank is less than 1 below the index value,
            # then assign the appropriate contribution to that index value
            ((left['__Index__'] - left['__Rank__'])
                                         .between(0,1,inclusive='neither'))
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
        left['__Contribution__'] = (
                left['__Proximity__'] * left[self.left_bound_col]
            )
        
        # If groupby_cols, then group data
        if groupby_cols == None:
            left = pd.DataFrame(
                [[max(left[self.left_boundary_col]),
                  sum(left['__Contribution__']),
                  min(left['__Contribution__'])]],
                columns=[
                    self.left_boundary_col,
                    self.left_bound_col,
                    '__Minimum__'
                    ]
                )
        # Determine left bound and boundary using the sum of the contributions
        # and an open boundary if any of the contributing values is open
        else:
            left = (
                left.groupby(groupby_cols)
                    .agg(**{
                        self.left_boundary_col: (self.left_boundary_col,
                                                 'max'),
                        self.left_bound_col: ('__Contribution__', 'sum'),
                        '__Minimum__': ('__Contribution__', 'min')
                        })
                    )
        
        # Replace the numeric value for the boundary
        left[self.left_boundary_col] = (
            left[self.left_boundary_col]
                .replace([0,1], ['Closed','Open'])
            )
        
        # Sums with infinite values produce nan values rather than 
        # np.inf values. Convert nan to inf only if infinite values are
        # included in the sum
        left[self.left_bound_col] = (
            np.where(left['__Minimum__'] == -np.inf,
                     -np.inf,
                     left[self.left_bound_col]
                     )
            )
        
        # Determine right bound and boundary for each group
        
        # Change notation of 'Closed' and 'Open' boundaries to 1 and 0,
        # respectively. Use 1 for closed to ensure that closed boundaries are
        # sorted larger than open boundaries when the right bound is tied
        df[self.right_boundary_col] = (
            df[self.right_boundary_col]
                .replace(['Closed','Open'], [1,0])
            )
        
        # Sort right bound values
        grouping = ['__Size__', self.right_boundary_col, self.right_bound_col]
        if groupby_cols != None:
            grouping = groupby_cols + grouping
        
        right = (
            df.copy()[grouping]
                .sort_values(
                    by = [self.right_bound_col, self.right_boundary_col]
                    )
            )
        
        # Add index for each group
        if groupby_cols == None:
            right['__Index__'] = range(1, 1+len(right))
        else:
            right['__Index__'] = right.groupby(groupby_cols).cumcount() + 1
        
        # Determine the rank in each group for the percentile
        right['__Rank__'] = round(
            C + percentile*(right['__Size__'] + 1 - 2*C),
            8
            )
        
        # Handle situations where rank is outside the range 1 to length of data
        conditions = [
            # Use maximum rank if rank is larger than the length of the dataset
            (right['__Rank__'] > right['__Size__']),
            # Use rank of 1 if rank is less than 1
            (right['__Rank__'] < 1)
            ]
        
        results = [
            right['__Size__'],
            1
            ]
        
        right['__Rank__'] = np.select(conditions, results, right['__Rank__'])
        
        # Generate proximity of each result to percentile rank using the index
        right['__Proximity__'] = 0
        
        conditions = [
            # If the percentile rank is a whole number,
            # then use that index result
            (right['__Rank__'] == right['__Index__']),
            # If the percentile rank is less than 1 above the index value,
            # then assign the appropriate contribution to that index value
            ((right['__Rank__'] - right['__Index__'])
                                         .between(0,1,inclusive='neither')),
            # If the percentile rank is less than 1 below the index value,
            # then assign the appropriate contribution to that index value
            ((right['__Index__'] - right['__Rank__'])
                                         .between(0,1,inclusive='neither'))
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
        right['__Contribution__'] = (
            right['__Proximity__'] * right[self.right_bound_col]
            )
        
        # If groupby_cols, then group data
        if groupby_cols == None:
            right = pd.DataFrame(
                [[sum(right['__Contribution__']),
                  min(right[self.right_boundary_col]),
                  max(right['__Contribution__'])]],
                columns=[
                    self.right_bound_col,
                    self.right_boundary_col,
                    '__Maximum__'
                    ]
                )
        # Determine left bound and boundary using the sum of the contributions
        # and an open boundary if any of the contributing values is open
        else:
            right = (
                right.groupby(groupby_cols)
                    .agg(**{
                        self.right_bound_col: ('__Contribution__', 'sum'),
                        self.right_boundary_col: (self.right_boundary_col,
                                                  'min'),
                        '__Maximum__': ('__Contribution__', 'max')
                        })
                    )
        
        # Replace the numeric value for the boundary
        right[self.right_boundary_col] = (
            right[self.right_boundary_col]
                .replace([1,0], ['Closed','Open'])
            )
        
        # Sums with infinite values produce nan values rather than
        # np.inf values. Convert nan to inf only if infinite values are
        # included in the sum
        right[self.right_bound_col] = (
            np.where(right['__Maximum__'] == np.inf,
                     np.inf,
                     right[self.right_bound_col]
                     )
            )
        
        if groupby_cols == None:
            df = df[['__Warning__']].drop_duplicates()
            df = pd.concat([left,right,df], axis=1)
        else:
            # Merge the two boundaries to create the interval for the percentile
            # Check that the merge is 1-to-1
            df = df[groupby_cols + ['__Warning__']].drop_duplicates()
            left = df.merge(left,
                            how = 'inner',
                            on = groupby_cols)
            df = left.merge(right,
                            how = 'outer',
                            on = groupby_cols,
                            validate = '1:1'
                            )
        
        # Create a column that indicates the generated statistic
        if percentile == 0.5:
            df[self.stat_col] = 'Median'
        else:
            df[self.stat_col] = f'Percentile-{100*percentile}'
        
        # Reorder columns
        if groupby_cols == None:
            df = df[[self.stat_col,
                     self.left_boundary_col,
                     self.left_bound_col,
                     self.right_bound_col,
                     self.right_boundary_col,
                     '__Warning__'
                     ]]
        else:
            df = df[groupby_cols +
                    [self.stat_col,
                     self.left_boundary_col,
                     self.left_bound_col,
                     self.right_bound_col,
                     self.right_boundary_col,
                     '__Warning__'
                     ]]
        
        # Reset index
        # df = df.reset_index()
        
        # Replace or return result
        if inplace:
            self.data = df
        else:
            return df
    
    #%% Exceedance assessment
    
    def determine_exceedances(self,
                              threshold,
                              threshold_is_exceedance = False,
                              inplace = False):
        
        # Check that threshold is integer or float
        if not isinstance(threshold, (int, float)):
            raise ValueError('Input for threshold needs to be integer or '
                             'float type.')
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Set conditions depending on whether the threshold is an exceedance
        if threshold_is_exceedance:
            # Set conditions for exceedances and non-exceedances
            conditions = [
                # Exceedance condition
                # The left bound is greater than or equal to the threshold
                (df[self.left_bound_col] >= threshold),
                # Non-exceedance conditions
                (
                    # The right bound is less than the threshold
                    (df[self.right_bound_col] < threshold) |
                    # The right bound is equal to the threshold and open
                    ((df[self.right_bound_col] == threshold) &
                     (df[self.right_boundary_col] == 'Open'))
                )
                ]
        # Set conditions when threshold is not an exceedance
        else:
            # Set conditions for exceedances and non-exceedances
            conditions = [
                # Exceedances conditions
                (
                    # The left bound is greater than the threshold
                    (df[self.left_bound_col] > threshold) |
                    # If the left bound is equal to the threshold and open
                    ((df[self.left_bound_col] == threshold) &
                     (df[self.left_boundary_col] == 'Open'))
                ),
                # Non-exceedance condition
                # The right bound is less than or equal to the threshold
                (df[self.right_bound_col] <= threshold)
                ]
        # Exceedances are indicated by integers
        # (1=exceedance, 0=non-exceedance)
        results = [
            1,
            0
            ]
        # Default exceedance (np.nan=unknown)
        df[self.exceedances_col] = np.select(conditions, results, np.nan)
        
        # Replace or return result
        if inplace:
            self.data = df
        else:
            return df
    
    #%% Statistic methods
    
    def general_stat(self,
                     stat,
                     groupby_cols = None,
                     count_cols = None,
                     inplace = False
                     ):
        
        # Create a copy of the CensoredData object
        cdf = copy.deepcopy(self)
        
        # If no groupby_cols provided, calculate stat for full dataset
        if groupby_cols == None:
            stat(cdf, [], inplace = True)
        # If groupby_cols is a list of lists, then calculate stat for each
        # sequential group
        elif isinstance(groupby_cols[0], list):
            for grouping in groupby_cols:
                stat(cdf, grouping, inplace = True)
        # If the groupby_cols is a list of column names, then calculate stat
        # for each group
        elif isinstance(groupby_cols, list):
            stat(cdf, groupby_cols, inplace = True)
        # Else unexpected format for groupby_cols
        else:
            raise ValueError('The value provided for "groupby_cols" must be a '
                             'list rather than {type(groupby_cols)}.')
        
        # Set focus of high or low
        if stat == CensoredData.maximum_interval:
            focus_high_potential = True
        elif stat == CensoredData.minimum_interval:
            focus_high_potential = False
        else:
            focus_high_potential = self.focus_high_potential
        
        # Convert the interval for the stat into censor and numeric notation
        cdf.components_from_interval(focus_high_potential, inplace = True)
        
        # Create column with interval notation
        cdf.interval_notation(inplace = True)
        
        # Combine the censor and numeric components into a result
        cdf.result_from_components(inplace = True)
        
        # Reorder columns
        output_cols = [cdf.stat_col,
                          cdf.result_col,
                          cdf.censor_col,
                          cdf.numeric_col,
                          cdf.interval_col]
        
        if groupby_cols == None:
            cdf.data = cdf.data[output_cols]
        elif (all(isinstance(name, str) for name in groupby_cols)):
            cdf.data = cdf.data[groupby_cols + output_cols]
        else:
            cdf.data = cdf.data[groupby_cols[-1] + output_cols]
        
        # Merge count info
        if count_cols != None:
            cdf.data = self.merge_counts(cdf.data,
                                         count_cols,
                                         groupby_cols)
        
        # Replace or return result
        if inplace:
            self.data = cdf.data
        else:
            return cdf.data
    
    def maximum(self,
                groupby_cols = None,
                count_cols = None,
                inplace = False):
        
        return self.general_stat(CensoredData.maximum_interval,
                                 groupby_cols,
                                 count_cols,
                                 inplace)
    
    def minimum(self,
                groupby_cols = None,
                count_cols = None,
                inplace = False):
        
        return self.general_stat(CensoredData.minimum_interval,
                                 groupby_cols,
                                 count_cols,
                                 inplace)
    
    def mean(self,
                groupby_cols = None,
                count_cols = None,
                inplace = False):
        
        return self.general_stat(CensoredData.mean_interval,
                                 groupby_cols,
                                 count_cols,
                                 inplace)
    
    def add(self,
                groupby_cols = None,
                count_cols = None,
                inplace = False):
        
        return self.general_stat(CensoredData.add_interval,
                                 groupby_cols,
                                 count_cols,
                                 inplace)
        
    def percentile(self,
                   groupby_cols = None,
                   count_cols = None,
                   percentile = None,
                   method = 'hazen',
                   inplace = False
                   ):
        
        # Create a copy of the CensoredData object
        cdf = copy.deepcopy(self)
        
        # Check for percentile
        if (percentile > 100) | (percentile < 0):
            raise ValueError('The percentile must be between 0 and 100.')
        
        # If no groupby_cols provided, calculate stat for full dataset
        if groupby_cols == None:
            cdf.percentile_interval(groupby_cols,
                                    percentile,
                                    method,
                                    inplace = True)
        # If groupby_cols is a list of lists, then calculate stat for each
        # sequential group, use median for all but last group
        elif isinstance(groupby_cols[0], list):
            for grouping in groupby_cols:
                if grouping == groupby_cols[-1]:
                    cdf.percentile_interval(grouping,
                                            percentile,
                                            method,
                                            inplace=True)
                else:
                    cdf.percentile_interval(grouping,
                                            50,
                                            method,
                                            inplace=True)
        # If the groupby_cols is a list of column names, then calculate stat
        # for each group
        elif isinstance(groupby_cols, list):
            cdf.percentile_interval(groupby_cols,
                                    percentile,
                                    method,
                                    inplace=True)
        # Else unexpected format for groupby_cols
        else:
            raise ValueError('The value provided for "groupby_cols" must be a '
                             'list rather than {type(groupby_cols)}.')
        
        # Convert the interval for the stat into censor and numeric notation
        cdf.components_from_interval(cdf.focus_high_potential, inplace = True)
        
        # Create column with interval notation
        cdf.interval_notation(inplace = True)
        
        # Combine the censor and numeric components into a result
        cdf.result_from_components(inplace = True)
        
        cdf.data[cdf.result_col] += cdf.data['__Warning__']
        
        # Reorder columns
        output_cols = [cdf.stat_col,
                          cdf.result_col,
                          cdf.censor_col,
                          cdf.numeric_col,
                          cdf.interval_col]
        
        if groupby_cols == None:
            cdf.data = cdf.data[output_cols]
        elif (all(isinstance(name, str) for name in groupby_cols)):
            cdf.data = cdf.data[groupby_cols + output_cols]
        else:
            cdf.data = cdf.data[groupby_cols[-1] + output_cols]
        
        # Merge count info
        if count_cols != None:
            cdf.data = self.merge_counts(cdf.data,
                                         count_cols,
                                         groupby_cols)
        
        # Replace or return result
        if inplace:
            self.data = cdf.data
        else:
            return cdf.data
    
    def median(self,
               groupby_cols = None,
               count_cols = None,
               inplace = False):
        
        return self.percentile(groupby_cols,
                               count_cols,
                               50,
                               inplace=inplace)
    
    def percent_exceedances(self,
                            threshold,
                            threshold_is_exceedance,
                            groupby_cols = None,
                            count_cols = None,
                            round_to = 2,
                            inplace = False):
        
        # Create a copy of the CensoredData object
        cdf = copy.deepcopy(self)
        
        # Determine exceedances for each sample
        cdf.determine_exceedances(threshold,
                                  threshold_is_exceedance,
                                  inplace = True)
        
        df = cdf.data
        
        # If no groupby_cols provided, calculate stat for full dataset
        if groupby_cols == None:
            df = (df
                .agg(**{
                    # Sum exceedances
                    cdf.exceedances_col: (cdf.exceedances_col,'sum'),
                    # Count of assessable values
                    '__DeterminedCount__': (cdf.exceedances_col,'count'),
                    # Count of all values (including nan)
                    '__TotalCount__': (cdf.exceedances_col,'size')
                    })
                # Flip series into a table
                .transpose()
                .reset_index(drop=True)
                )
        # If groupby_cols is a list of lists, then assess maximums for first
        # grouping and percentage of exceedances for the second grouping
        elif (isinstance(groupby_cols[0],list)) & (len(groupby_cols) != 2):
            raise ValueError('Only two lists can be included for groupby_cols '
                             'that is a list of lists. Review the '
                             'documentation for this method.')
        elif isinstance(groupby_cols[0],list):
            # Use maximum within each group for first grouping
            df = (df.groupby(groupby_cols[0])[cdf.exceedances_col]
                      .max()
                      .reset_index())
            # Use second grouping for percentage of exceedances
            df = (df.groupby(groupby_cols[1])
                  .agg(**{
                      cdf.exceedances_col: (cdf.exceedances_col,'sum'),
                      '__DeterminedCount__': (cdf.exceedances_col,'count'),
                      '__TotalCount__': (cdf.exceedances_col,'size')
                    })
                )
        # If the groupby_cols is a list of column names, then calculate stat
        # for each group
        elif isinstance(groupby_cols, list):
            df = (df.groupby(groupby_cols)
                  .agg(**{
                      cdf.exceedances_col: (cdf.exceedances_col,'sum'),
                      '__DeterminedCount__': (cdf.exceedances_col,'count'),
                      '__TotalCount__': (cdf.exceedances_col,'size')
                    })
                )
        # Else unexpected format for groupby_cols
        else:
            raise ValueError('The value provided for "groupby_cols" must be a '
                             'list rather than {type(groupby_cols)}.')
        
        # Determine counts and calculate percentage
        df[cdf.non_exceedances_col] = (
            df['__DeterminedCount__'] - df[cdf.exceedances_col]
            )
        df[cdf.ignored_col] = df['__TotalCount__'] - df['__DeterminedCount__']
        df[cdf.numeric_col] = (
            round(df[cdf.exceedances_col] / df['__DeterminedCount__'] * 100,
                  round_to)
            )
        df[cdf.result_col] = df[cdf.numeric_col].astype(str) + '%'
        df.loc[df[cdf.ignored_col] > 0, cdf.result_col] = (
            df[cdf.result_col] +
            ' (' + df[cdf.ignored_col].astype(str) + ' value(s) ignored)'
            )
        
        # Create a column that indicates the generated statistic 
        df[cdf.stat_col] = 'Percent Exceedance'
        df[cdf.threshold_col] = f'{threshold}'
        if threshold_is_exceedance:
            df[cdf.threshold_col] = '<' + df[cdf.threshold_col]
        
        # Reset index
        df = df.reset_index()
        
        # Reorder columns
        output_cols = [cdf.stat_col,
                       cdf.threshold_col,
                       cdf.result_col,
                       cdf.numeric_col,
                       cdf.exceedances_col,
                       cdf.non_exceedances_col,
                       cdf.ignored_col]
        
        if groupby_cols == None:
            cdf.data = df[output_cols]
        elif (all(isinstance(name, str) for name in groupby_cols)):
            cdf.data = df[groupby_cols + output_cols]
        else:
            cdf.data = df[groupby_cols[-1] + output_cols]
        
        # Merge count info
        if count_cols != None:
            cdf.data = self.merge_counts(cdf.data,
                                         count_cols,
                                         groupby_cols)
        
        # Replace or return result
        if inplace:
            self.data = cdf.data
        else:
            return cdf.data
