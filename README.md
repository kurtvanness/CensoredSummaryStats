# censoredsummarystats
A repository that contains a CensoredData class for analyzing censored data for basic stats.

## Class settings:

When initiating the class, there are two required inputs, 6 default analysis settings that can be changed, and several output column names that can be customized.

Required inputs:

- **data**: A pandas DataFrame containing data
- **value_col**: The column name for the column of potentially censored values

Default analysis settings:

- **include_negative_interval**: (default False) This setting controls whether left censored results are assumed to have a lower bound of 0 or whether the result can be negative.
- **focus_high_potential**: (default True) This setting controls whether the highest potential or lowest potential is the focus for the result. The functions in this repository consider the full range of potential values of a censored result. This can often lead to a potential range for a statistical result. This setting determines whether to focus on the high or low end of the range. The maximum and minimum statistic ignores this setting and focuses on the high and low end of possible values, respectively.
- **precision_tolerance_to_drop_censor**: (default 0.25) This setting controls whether a range of possible results is returned as censored or non-censored. For example, if an average is known to be between 2 and 3, then 2.5 would be returned as the result since the whole range is within 20% of 2.5 (20% < 25%). If this parameter was set to 0.15 (15%), then the tolerance would not cover the range and a result of <3 or >2 would be returned depending on the value of focus_high_potential.
- **precision_rounding**: (default True) This setting controls whether to apply a specific rounding procedure (discussed below).
- **thousands_comma**: (default False) This setting controls whether to output values over 1000 with commas (1,000).
- **output_interval**: (default True) This setting controls whether to output the interval was converted to the output result.

Customizable output column names:

- **stat_col**: (default 'Statistic') This column contains the statatistic that was analyzed (minimum, maximum, median, etc.).
- **result_col**: (default 'Result') This column contains the result of the statistical analysis as a string value. It may contain additional information for percentile or percent exceedance results.
- **censor_col**: (default 'CensorComponent') This column contains the censor component for statistical results.
- **numeric_col**: (default 'NumericComponent') This column contains the numeric component for statistical results.
- **interval_col**: (default 'Interval') This column contains the possible range of the statistical result when considering all possibilities of censored values.
- **threshold_col**: (default 'Threshold') This column contains the threshold supplied by the user for the percent_exceedances method.
- **exceedances_col**: (default 'Exceedances') This column contains the number of exceedances resulting from the percent_exceedances method.
- **non_exceedances_col**: (default 'NonExceedances') This column contains the number of non-exceedances resulting from the percent_exceedances method.
- **ignored_col**: (default 'Ignored') This column contains the number of values that couldn't be assessed for the percent_exceedances method. For example, a value of '<2' cannot be determined as being above or below 1. Users should manually adjust values to be above or below the supplied threshold if they need to be considered.

## Precision Rounding approach:

There is a built in rounding option that is based on common water quality reporting measurement precision. 

| NumericComponent  | Rounding process |
| ------------- | ------------- |
| ≥100  | 3 significant figures  |
| ≥10 to <100  | 1 decimal place  |
| ≥0.2 to <10  | 2 decimal place  |
| ≥0.1 to <0.2  | 3 decimal place  |
| <0.1  | 2 significant figures  |

## Methods
The primary methods include maximum, minimum, mean, percentile, median (same as 50th percentile). Additional methods are described below.

The class requires a dataframe that contains a column of potentially censored values. This package is most useful when the results are written as strings that cannot be directly converted to a numeric datatype due to the presence of symbols that indicate the value is potentially above or below a particular value. The accepted censorship symbols include (<,≤,≥,>).

Additional table columns can be provided as a list so that the statistical functions obtain results for specified groups.

1.	**Maximum**: Calculate the maximum value for a set of values.

2.	**Minimum**: Calculate the minimum value for a set of values.

3.	**Mean/Average**: Calculate the average value for a set of values.

4.	**Median**: Calculate the median value for a set of values.

5.	**Percentile**: Calculate a percentile for a set of values. The desired percentile should be provided as a number between 0 and 100. The default percentile method is Hazen, but other methods include Weibull, Tukey, Blom, and Excel as described in https://environment.govt.nz/assets/Publications/Files/hazen-percentile-calculator-2.xls

6.	**Addition**: Calculate the sum for a set of values.

7.	**Percent Exceedances**: Calculate the percentage of values that exceed a specified threshold. The desired threshold should be provided as a number. The default is to not treat results equal to the threshold as exceedances, but this can be changed by setting threshold_is_exceedance to True.


## Method settings:

Many of the methods above have similar input parameters. Those are:

- **groupby_cols**: (default None) These are the columns that should be used to define the groups. Multiple groupings can be provided for some functions. This is useful to even weight data over sites or time periods. For example, a potential input for could be [['Year','Month','Day'], ['Year','Month'], ['Year']]. This would ensure that all days are evenly weighted within the month and that all months are evenly weighted within the year for a stat such as mean or median.
- **count_cols**: (default None) Supplying a list of strings here will cause methods to return value counts. There should be the same number of strings as there are groupings in groupby_cols. Using the same example for groupby_cols, a user could supply ['Samples', 'Days Sampled', 'Months Sampled'] to get value counts for each grouping.
- **stat_name**: (default is statistic) The text to use to describe the stat ('Minimum', 'Median', etc.)
- **filters**: (default None) A dictionary of column names with values to filter for. This allows some simple filtering without recreating CensoredData objects.

## Dependencies

For the installation of `censoredsummarystats`, the following packages are required:
- [numpy >= 1.24](https://www.numpy.org/)

## Installation

You can install `censoredsummarystats` using pip. For Windows users

```python
pip install censoredsummarystats
```

## Usage

A quick example of `censoredsummarystats` usage is given below.

```python
import pandas as pd
import censoredsummarystats as css

# Create DataFrame
df = pd.DataFrame([['Set1',1.5],['Set1','2'],['Set1','>2.5'],['Set2',2],['Set2','<3'],['Set2',7.0]],columns=['Groups','Results'])

Groups Results
0   Set1     1.5
1   Set1       2
2   Set1    >2.5
3   Set2       2
4   Set2      <3
5   Set2     7.0

# Create CensoredData object from dataframe
cdf = css.CensoredData(data=df,value_col='Results')

# Calculate minimums, averages, and medians for the data
minimums = cdf.minimum(groupby_cols=['Groups'])

averages = cdf.mean(groupby_cols=['Groups'])

medians = cdf.median(groupby_cols=['Groups'])
```
Output are like this:
```python
print(minimums)
  Groups Statistic Result CensorComponent  NumericComponent      Interval
0   Set1   Minimum   1.50                               1.5  [1.50, 1.50]
1   Set2   Minimum  ≤2.00               ≤               2.0     [0, 2.00]

print(averages)
  Groups Statistic Result CensorComponent  NumericComponent      Interval
0   Set1      Mean  >2.00               >               2.0   (2.00, inf)
1   Set2      Mean   3.50                               3.5  [3.00, 4.00)

print(medians)
  Groups Statistic Result CensorComponent  NumericComponent      Interval
0   Set1    Median   2.00                               2.0  [2.00, 2.00]
1   Set2    Median   2.50                               2.5  [2.00, 3.00)
```

