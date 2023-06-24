
import numpy as np
import pandas as pd
from dataclasses import dataclass

from censoredsummarystats.validation import _validate_cdf, _validate_groupby_cols
from censoredsummarystats.stat_interval_aggregation import (
    _maximum_interval,
    _minimum_interval,
    _mean_interval,
    _sum_interval,
    _percentile_interval,
    _median_interval)
from censoredsummarystats.interval_to_result import _interval_to_result
from censoredsummarystats.percent_exceedance import (
    _determine_exceedances,
    _group_exceedances,
    _percent_exceedances)
from censoredsummarystats.merge_count_info import _merge_count_info

@dataclass
class CensoredData:
    data: pd.core.frame.DataFrame
    value_col: any
    include_negative_interval: bool = False
    focus_high_potential: bool = True
    precision_tolerance_to_drop_censor: float = 0.25
    precision_rounding: bool = True
    thousands_comma: bool = False
    output_interval: bool = True
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
    ignored_col: str = 'IgnoredValues'
    warning_col: str = 'Warning'
    
    def __post_init__(self):
        
        #%% Check data input
        
        _validate_cdf(self)
        
        #%% Split value into censor/numeric components
        
        # Create temporary column for first character
        first_character = self.data[self.value_col].astype(str).str[0]
        # Determine if the value has a censor component
        is_censored = first_character.isin(['<','≤','≥','>'])
        
        # Create censor column using first characters that are censors
        self.data[self.censor_col] = (
            np.where(
                is_censored,
                first_character,
                ''
                )
            )
        
        # Create numeric column using left remaining part of value
        self.data[self.numeric_col] = (
            np.where(
                is_censored,
                self.data[self.value_col].astype(str).str[1:],
                self.data[self.value_col]
                )
            )
        
        # Convert numeric component to float
        try:
            self.data[self.numeric_col] = (self.data[self.numeric_col]
                                               .astype(float))
        except ValueError:
            raise Exception('At least one value could not be converted to a '
                'numeric data type. Check that each value has no more than a '
                'single censor character that is one of the following: '
                '<, ≤, ≥, >')
            
        
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
                    'negative values or set include_negative_interval = True')
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
    
    #%% Statistic methods
    
    def _general_stat(self,
                      stat,
                      groupby_cols = None,
                      count_cols = None,
                      stat_name = None,
                      filters = None):
        
        # Validate groupby_cols
        _validate_groupby_cols(self, groupby_cols)
        
        # Create a copy of the CensoredData object
        df = self.data.copy()
        
        # Apply filters
        if filters:
            for col_name in filters:
                df = df[df[col_name].isin(filters[col_name])]
        
        # If no groupby_cols provided, calculate stat for full dataset
        if groupby_cols == None:
            df = stat(self, df, None, stat_name)
        else:
            for grouping in groupby_cols:
                df = stat(self, df, grouping, stat_name)
        
        # Set focus of high or low
        if stat == _maximum_interval:
            focus_high_potential = True
        elif stat == _minimum_interval:
            focus_high_potential = False
        else:
            focus_high_potential = self.focus_high_potential
        
        # Create result from interval information
        df = _interval_to_result(self, df, focus_high_potential)
        
        # Merge count info
        if count_cols:
            df = _merge_count_info(self, df, groupby_cols, count_cols, filters)
        
        return df
    
    def maximum(self,
                groupby_cols = None,
                count_cols = None,
                stat_name = 'Maximum',
                filters = None):
        
        return self._general_stat(_maximum_interval,
                                  groupby_cols,
                                  count_cols,
                                  stat_name,
                                  filters)
    
    def minimum(self,
                groupby_cols = None,
                count_cols = None,
                stat_name = 'Minimum',
                filters = None):
        
        return self._general_stat(_minimum_interval,
                                  groupby_cols,
                                  count_cols,
                                  stat_name,
                                  filters)
    
    def mean(self,
             groupby_cols = None,
             count_cols = None,
             stat_name = 'Mean',
             filters = None):
        
        return self._general_stat(_mean_interval,
                                  groupby_cols,
                                  count_cols,
                                  stat_name,
                                  filters)
    
    def average(self,
                groupby_cols = None,
                count_cols = None,
                stat_name = 'Average',
                filters = None):
        
        return self._general_stat(_mean_interval,
                                  groupby_cols,
                                  count_cols,
                                  stat_name,
                                  filters)
    
    def add(self,
            groupby_cols = None,
            count_cols = None,
            stat_name = 'Sum',
            filters = None):
        
        return self._general_stat(_sum_interval,
                                  groupby_cols,
                                  count_cols,
                                  stat_name,
                                  filters)
    
    def median(self,
               groupby_cols = None,
               count_cols = None,
               stat_name = 'Median',
               filters = None):
        
        return self._general_stat(_median_interval,
                                  groupby_cols,
                                  count_cols,
                                  stat_name,
                                  filters)
        
    def percentile(self,
                   percentile,
                   groupby_cols = None,
                   count_cols = None,
                   stat_name = 'Percentile',
                   method = 'hazen',
                   filters = None):
        
        # Create a copy of the CensoredData object
        df = self.data.copy()
        
        # Apply filters
        if filters:
            for col_name in filters:
                df = df[df[col_name].isin(filters[col_name])]
        
        # Validate percentile
        if (percentile > 100) | (percentile < 0):
            raise ValueError('The percentile must be between 0 and 100.')
        
        # Validate groupby_cols
        _validate_groupby_cols(self, groupby_cols)
        
        # If no groupby_cols provided, calculate percentile for full dataset
        if groupby_cols == None:
            df = _percentile_interval(self, df, percentile,
                                      None, stat_name, method)
        else:
            # Use median for all but the last group
            for grouping in groupby_cols:
                if grouping == groupby_cols[-1]:
                    df = _percentile_interval(self, df, percentile,
                                              grouping, stat_name, method)
                else:
                    df = _median_interval(self, df, grouping, stat_name)
        
        # Create result from interval information
        df = _interval_to_result(self, df)
        
        # Append warning message to result
        df[self.result_col] += ' ' + df[self.warning_col]
        
        # Drop warning column
        df = df.drop(columns=[self.warning_col])
        
        # Merge count info
        if count_cols:
            df = _merge_count_info(self, df, groupby_cols, count_cols, filters)
        
        return df
    
    def percent_exceedance(self,
                           threshold,
                           threshold_is_exceedance,
                           groupby_cols = None,
                           count_cols = None,
                           stat_name = 'Percent Exceedance',
                           round_to = 2,
                           filters = None):
        
        # Create a copy of the CensoredData object
        df = self.data.copy()
        
        # Apply filters
        if filters:
            for col_name in filters:
                df = df[df[col_name].isin(filters[col_name])]
        
        # Validate groupby_cols
        _validate_groupby_cols(self, groupby_cols)
        
        # Validate round_to input
        if (isinstance(round_to, bool) |
                (not isinstance(round_to, int))):
            raise ValueError(f'The value supplied to round_to must be an '
                'integer. Instead, the value supplied was '
                f'{repr(round_to)}.')
        
        # Assess the data for exceedances
        df = _determine_exceedances(self, df, threshold, threshold_is_exceedance)
        
        # Create stat column
        df[self.stat_col] = stat_name
        
        # Create empty list if no groups
        if groupby_cols == None:
            groupby_cols = []
        else:
            if len(groupby_cols) > 2:
                raise ValueError('The percent exceedance statistic can only '
                    'receive two lists for groupby_cols yet '
                    f'{len(groupby_cols)} were passed.')
        
        if len(groupby_cols) == 2:
            # Use maximum within each group for first grouping
            df = _group_exceedances(self, df, groupby_cols[0]+[self.stat_col])
        
        # Count exceedances
        if groupby_cols == None:
            df = _percent_exceedances(self, df,
                                      threshold, threshold_is_exceedance,
                                      [self.stat_col],
                                      round_to)
        else:
            df = _percent_exceedances(self, df,
                                      threshold, threshold_is_exceedance,
                                      groupby_cols[-1]+[self.stat_col],
                                      round_to)
        
        # Reorder columns
        df[self.result_col] = df.pop(self.result_col)
        df[self.numeric_col] = df.pop(self.numeric_col)
        df[self.exceedances_col] = df.pop(self.exceedances_col)
        df[self.non_exceedances_col] = df.pop(self.non_exceedances_col)
        df[self.ignored_col] = df.pop(self.ignored_col)
        
        # Merge count info
        if count_cols:
            df = _merge_count_info(self, df, groupby_cols, count_cols, filters)
        
        return df
