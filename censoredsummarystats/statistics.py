
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from censoredsummarystats.validation import (
    _validate_cdf,
    _validate_groupby_cols)
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
    '''
    This dataclass validates and pre-processes a dataframe for generating
    summary stats on censored data.

    Parameters
    ----------
    data : DataFrame
        A pandas dataframe containing censored data
    value_col : string
        The column name for the column with censored results
    include_negative_potential : bool
        This setting indicates whether negative values are expected or not.
        The default is false.
    focus_high_potential : bool
        This setting indicates whether the highest or lowest potential
        stat result should be focused. The default is true.
    precision_tolerance_to_drop_censor : float
        When the statistic results in a potential range of values, the midpoint
        is used as the result if all values are within the precision specified
        by this setting. The default is 0.25 (25%).
    precision_rounding : bool
        This setting indicates whether a specified rounding technique should
        be used. The default is true.
    thousands_comma : bool
        This setting indicates whether results larger than 1000 should include
        thousands commas or not. The default is false.
    output_interval : bool
        This setting indicates whether a column should be included in the
        output with the interval notation used to generate the result.
        The default is true.
    '''
    
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
    _minimum_col: str = field(default='__tempMinimum__', repr=False)
    _maximum_col: str = field(default='__tempMaximum__', repr=False)
    _size_col: str = field(default='__tempSize__', repr=False)
    _rank_col: str = field(default='__tempRank__', repr=False)
    _index_col: str = field(default='__tempIndex__', repr=False)
    _proximity_col: str = field(default='__tempProximity__', repr=False)
    _contribution_col: str = field(default='__tempContribution__', repr=False)
    _midpoint_col: str = field(default='__tempMidpointValue__', repr=False)
    _determined_col: str = field(default='__tempDeterminedCount__', repr=False)
    _totalcount_col: str = field(default='__tempTotalCount__', repr=False)
    
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
                      stat_function,
                      groupby_cols = None,
                      count_cols = None,
                      stat_name = None,
                      filters = None):
        
        # Validate groupby_cols
        _validate_groupby_cols(self, groupby_cols)
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Apply filters
        if filters:
            for col_name in filters:
                df = df[df[col_name].isin(filters[col_name])]
        
        # If no groupby_cols provided, calculate stat for full dataset
        if groupby_cols == None:
            df = stat_function(self, df, None, stat_name)
        else:
            for grouping in groupby_cols:
                df = stat_function(self, df, grouping, stat_name)
        
        # Set focus of high or low
        if stat_function == _maximum_interval:
            focus_high_potential = True
        elif stat_function == _minimum_interval:
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
        '''
        Return a dataframe with maximum values for a dataset

        Parameters
        ----------
        groupby_cols : list of list of strings
            These are the columns that should be used to define the groups. The
            ability to provide multiple lists is useful for generating counts.
            The default is None.
        count_cols : list of strings
            The column names to use for counts given to the above groups.
            The default is None.
        stat_name : str
            The name to use to describe the stat. The default is 'Maximum'.
        filters : dictionary of keywords with lists
            This parameter allows users to determine the stat for only
            specified values within columns. The default is None.

        Returns
        -------
        DataFrame
            A Dataframe of maximums for the specified groups.

        '''
        
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
        '''
        Return a dataframe with minimum values for a dataset

        Parameters
        ----------
        groupby_cols : list of list of strings
            These are the columns that should be used to define the groups. The
            ability to provide multiple lists is useful for generating counts.
            The default is None.
        count_cols : list of strings
            The column names to use for counts given to the above groups.
            The default is None.
        stat_name : str
            The name to use to describe the stat. The default is 'Minimum'.
        filters : dictionary of keywords with lists
            This parameter allows users to determine the stat for only
            specified values within columns. The default is None.

        Returns
        -------
        DataFrame
            A Dataframe of minimums for the specified groups.

        '''
        
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
        '''
        Return a dataframe with mean values for a dataset

        Parameters
        ----------
        groupby_cols : list of list of strings
            These are the columns that should be used to define the groups. The
            ability to provide multiple lists is useful for ensuring even
            weighting within subgroups or for generating counts of subgroups.
            The default is None.
        count_cols : list of strings
            The column names to use for counts given to the above groups.
            The default is None.
        stat_name : str
            The name to use to describe the stat. The default is 'Mean'.
        filters : dictionary of keywords with list of specified values
            This parameter allows users to determine the stat for only
            specified values within columns. The default is None.

        Returns
        -------
        DataFrame
            A Dataframe of means for the specified groups.

        '''
        
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
        '''
        Return a dataframe with average values for a dataset

        Parameters
        ----------
        groupby_cols : list of list of strings
            These are the columns that should be used to define the groups. The
            ability to provide multiple lists is useful for ensuring even
            weighting within subgroups or for generating counts of subgroups.
            The default is None.
        count_cols : list of strings
            The column names to use for counts given to the above groups.
            The default is None.
        stat_name : str
            The name to use to describe the stat. The default is 'Average'.
        filters : dictionary of keywords with list of specified values
            This parameter allows users to determine the stat for only
            specified values within columns. The default is None.

        Returns
        -------
        DataFrame
            A Dataframe of averages for the specified groups.

        '''
        
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
        '''
        Return a dataframe with sums for a dataset

        Parameters
        ----------
        groupby_cols : list of list of strings
            These are the columns that should be used to define the groups. The
            ability to provide multiple lists is useful for generating counts 
            of subgroups. The default is None.
        count_cols : list of strings
            The column names to use for counts given to the above groups.
            The default is None.
        stat_name : str
            The name to use to describe the stat. The default is 'Sum'.
        filters : dictionary of keywords with list of specified values
            This parameter allows users to determine the stat for only
            specified values within columns. The default is None.

        Returns
        -------
        DataFrame
            A Dataframe of sums for the specified groups.

        '''
        
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
        
        '''
        Return a dataframe with median values for a dataset

        Parameters
        ----------
        groupby_cols : list of list of strings
            These are the columns that should be used to define the groups. The
            ability to provide multiple lists is useful for ensuring even
            weighting within subgroups or for generating counts of subgroups.
            The default is None.
        count_cols : list of strings
            The column names to use for counts given to the above groups.
            The default is None.
        stat_name : str
            The name to use to describe the stat. The default is 'Median'.
        filters : dictionary of keywords with list of specified values
            This parameter allows users to determine the stat for only
            specified values within columns. The default is None.

        Returns
        -------
        DataFrame
            A Dataframe of medians for the specified groups.

        '''
        
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
        '''
        Return a dataframe with percentile values for a dataset

        Parameters
        ----------
        percentile : float
            The percentile that should be determined. The supplied value
            needs to be between 0 and 100.
        groupby_cols : list of list of strings
            These are the columns that should be used to define the groups. The
            ability to provide multiple lists is useful for ensuring even
            weighting within subgroups or for generating counts of subgroups.
            If subgroups are provided (multiple tiers of groupings) then the
            median values of the subgroups are determined before the percentile
            is determined for the final group.
            The default is None.
        count_cols : list of strings
            The column names to use for counts given to the above groups.
            The default is None.
        stat_name : str
            The name to use to describe the stat. The default is 'Percentile'.
        method : str
            The percentile method to use. Options include: 'weibull', 'tukey',
            'blom', 'hazen', 'excel'. These options come from the Ministry for
            the environment Hazen percentile calculator spreadsheet.
            The default is hazen.
        filters : dictionary of keywords with list of specified values
            This parameter allows users to determine the stat for only
            specified values within columns. The default is None.

        Returns
        -------
        DataFrame
            A Dataframe of percentiles for the specified groups.

        '''
        
        # Create a copy of the data
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
                           threshold_is_exceedance = False,
                           groupby_cols = None,
                           count_cols = None,
                           stat_name = 'Percent Exceedance',
                           round_to = 2,
                           filters = None):
        '''
        Return a DataFrame with the percentage of results that exceed a
        specified threshold.

        Parameters
        ----------
        threshold : float or int
            The threshold value of interest.
        threshold_is_exceedance : bool
            Set whether the threshold value itself is considered an exceedance.
            The default is false.
        groupby_cols : list of list of strings
            These are the columns that should be used to define the groups. The
            ability to provide multiple lists is useful for ensuring even
            weighting within subgroups or for generating counts of subgroups.
            The default is None.
        count_cols : list of strings
            The column names to use for counts given to the above groups.
            The default is None.
        stat_name : str
            The name to use to describe the stat.
            The default is 'Percent Exceedance'.
        round_to : int
            The number of decimal places to round the percentage to.
            The default is 2.
        filters : dictionary of keywords with list of specified values
            This parameter allows users to determine the stat for only
            specified values within columns. The default is None.

        Returns
        -------
        DataFrame
            A Dataframe of percentage of exceedances for the specified groups.

        '''
        
        # Create a copy of the data
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
        df = _determine_exceedances(self, df, threshold,
                                    threshold_is_exceedance)
        
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
