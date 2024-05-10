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

- **stat_col**: (default 'Statistic') This column contains the statistic that was analyzed (minimum, maximum, median, etc.).
- **result_col**: (default 'Result') This column contains the result of the statistical analysis as a string value. It may contain additional information for percentile or percent exceedance results.
- **censor_col**: (default 'CensorComponent') This column contains the censor component for statistical results.
- **numeric_col**: (default 'NumericComponent') This column contains the numeric component for statistical results.
- **interval_col**: (default 'Interval') This column contains the possible range of the statistical result when considering all possibilities of censored values.
- **threshold_col**: (default 'Threshold') This column contains the threshold supplied by the user for the percent_exceedances method.
- **exceedances_col**: (default 'Exceedances') This column contains the number of exceedances resulting from the percent_exceedances method.
- **non_exceedances_col**: (default 'NonExceedances') This column contains the number of non-exceedances resulting from the percent_exceedances method.
- **ignored_col**: (default 'Ignored') This column contains the number of values that couldn't be assessed for the percent_exceedances method. For example, a value of '<2' cannot be determined as being above or below 1. Users should manually adjust values to be above or below the supplied threshold if they need to be considered.
- **warning_col**: (default 'Warning') This column is only generated for percentiles under certain situations.

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

1.	**maximum**: Calculate the maximum value for a set of values.

2.	**minimum**: Calculate the minimum value for a set of values.

3.	**mean** or **average**: Calculate the average value for a set of values.

4.	**median**: Calculate the median value for a set of values.

5.	**percentile**: Calculate a percentile for a set of values. The desired percentile should be provided as a number between 0 and 100. The default percentile method is Hazen, but other methods include Weibull, Tukey, Blom, and Excel as described in https://environment.govt.nz/assets/Publications/Files/hazen-percentile-calculator-2.xls

6.	**add**: Calculate the sum for a set of values.

7.	**percent_exceedance**: Calculate the percentage of values that exceed a specified threshold. The desired threshold should be provided as a number. The default is to not treat results equal to the threshold as exceedances, but this can be changed by setting threshold_is_exceedance to True.


## Method settings:

Many of the methods above have similar input parameters. Those are:

- **groupby_cols**: (default None) These are the columns that should be used to define the groups. Multiple groupings can be provided for some functions. This is useful to even weight data over sites or time periods. For example, a potential input for could be [['Year','Month','Day'], ['Year','Month'], ['Year']]. This would ensure that all days are evenly weighted within the month and that all months are evenly weighted within the year for a stat such as mean or median.
- **count_cols**: (default None) Supplying a list of strings here will cause methods to return value counts. There should be the same number of strings as there are groupings in groupby_cols. Using the same example for groupby_cols, a user could supply ['Samples', 'Days Sampled', 'Months Sampled'] to get value counts for each grouping.
- **stat_name**: (default is statistic) The text to use to describe the stat ('Minimum', 'Median', etc.)
- **filters**: (default None) A dictionary of column names with values to filter for. This allows some simple filtering without recreating CensoredData objects.

The percentile function has the following additional input parameters:
- **percentile**: The percentile that should be determined. The supplied value needs to be between 0 and 100.
- **method**: (default hazen) The percentile method to use. Options include: 'weibull', 'tukey', 'blom', 'hazen', 'excel'.

## Dependencies

For the installation of `censoredsummarystats`, the following packages are required:
- [numpy >= 1.24](https://www.numpy.org/)

## Installation

You can install `censoredsummarystats` using pip. For Windows users

```python
pip install censoredsummarystats
```

## Usage

An example of `censoredsummarystats` usage is given below.

```python
import pandas as pd
import censoredsummarystats as css

# Create DataFrame
df = pd.DataFrame([
    ['Site1','E. coli', 2021, '<1'],
    ['Site1','E. coli', 2021, '<1'],
    ['Site1','E. coli', 2022, 455],
    ['Site1','E. coli', 2022, '>2420'],
    ['Site1','E. coli', 2022, 257],
    ['Site2','E. coli', 2021, 5],
    ['Site2','E. coli', 2021, '>2420'],
    ['Site2','E. coli', 2022, '<10'],
    ['Site2','E. coli', 2022, 17000],
    ['Site2','Temperature', 2022, 12.4]
    ],
    columns=['SiteID','Parameter','Year','Result'])

  SiteID    Parameter  Year Result
0  Site1      E. coli  2021     <1
1  Site1      E. coli  2021     <1
2  Site1      E. coli  2022    455
3  Site1      E. coli  2022  >2420
4  Site1      E. coli  2022    257
5  Site2      E. coli  2021      5
6  Site2      E. coli  2021  >2420
7  Site2      E. coli  2022    <10
8  Site2      E. coli  2022  17000
9  Site2  Temperature  2022   12.4

# Create CensoredData object from dataframe
cdf = css.CensoredData(data=df,value_col='Result')

# Calculate annual maximums
annual_maximums = cdf.maximum(groupby_cols=[['SiteID','Parameter','Year']],
                                count_cols=['Samples'],
                                stat_name='Annual Maximum')

# Calculate the maximum concentrations of E. coli measured at each site
site_ecoli_maximums = cdf.maximum(groupby_cols=[['SiteID','Year'],['SiteID']],
                                count_cols=['Samples','YearsSampled'],
                                stat_name='Site Maximum',
                                filters = {'Parameter':['E. coli']})

# Calculate the annual averages
annual_averages = cdf.average(groupby_cols=[['SiteID','Parameter','Year']],
                                count_cols=['Samples'],
                                stat_name='Annual Average')

```
Outputs are like this:
```python
print(annual_maximums)
	SiteID	Parameter	Year	Statistic	Result	Interval	Samples
0	Site1	E. coli	        2021	Annual Maximum	<1	[0, 1)	        2
1	Site1	E. coli	        2022	Annual Maximum	>2420	(2420, inf)	3
2	Site2	E. coli	        2021	Annual Maximum	>2420	(2420, inf)	2
3	Site2	E. coli	        2022	Annual Maximum	17000	[17000, 17000]	2
4	Site2	Temperature	2022	Annual Maximum	12.4    [12.4, 12.4]	1


print(site_ecoli_maximums)
	SiteID	Statistic	Result	Interval	Samples	YearsSampled
0	Site1	Site Maximum	>2420	(2420, inf)	5	2
1	Site2	Site Maximum	≥17000	[17000, inf)	4	2


print(annual_averages)
	SiteID	Parameter	Year	Statistic	Result	Interval	Samples
0	Site1	E. coli	        2021	Annual Average	<1	[0, 1)	        2
1	Site1	E. coli	        2022	Annual Average	>1040	(1044, inf)	3
2	Site2	E. coli	        2021	Annual Average	>1210	(1212.5, inf)	2
3	Site2	E. coli	        2022	Annual Average	8500	[8500, 8505)	2
4	Site2	Temperature	2022	Annual Average	12.4	[12.4, 12.4]	1
```

