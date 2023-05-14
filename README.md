# censoredsummarystats
A repository that contains functions for calculating summary stats on censored data.

## Functions
The primary functions include maximum, minimum, average, percentile, median (same as 50th percentile). Additional functions are described below.

All functions require a dataframe that contains a column of results. This package is most useful when the results are written as strings that cannot be directly converted to a numeric datatype due to the presence of symbols that indicate the result is potentially above or below a particular value. The accepted censorship symbols include (<,≤,≥,>).

Additional table columns can be provided as a list so that the statistical functions obtain results for specified groups.

1.	**Maximum**: Calculate the maximum value for a set of results.

2.	**Minimum**: Calculate the minimum value for a set of results.

3.	**Average**: Calculate the average value for a set of results.

4.	**Median**: Calculate the median value for a set of results.

5.	**Percentile**: Calculate a percentile for a set of results. The desired percentile should be provided as a number between 0 and 100. The default percentile method is Hazen, but other methods include Weibull, Tukey, Blom, and Excel as described in https://environment.govt.nz/assets/Publications/Files/hazen-percentile-calculator-2.xls

6.	**Addition**: Calculate the sum for a set of results.

7.	**Percent Exceedances**: Calculate the percentage of results that exceed a specified threshold. The desired threshold should be provided as a number. The default is to not treat results equal to the threshold as exceedances, but this can be changed by setting count_threshold_as_exceedance to True.

8.	**Results to Components**: Split results into censor component (text) and numeric component (number).

## Function settings:

Many of the functions above have similar input parameters. Those are:

- **df**: A pandas DataFrame containing data
- **result_col**: The column name for the column of results
- **groupby_cols**: The columns that should be used to define the groups. Multiple groupings can be provided for some functions.
- **focus_high_potential**: (default True) This setting controls whether the highest potential or lowest potential is the focus for the result. The functions in this repository consider the full range of potential values of a censored result. This can often lead to a potential range for a statistical result. This setting determines whether to focus on the high or low end of the range.
- **include_negative_interval**: (default False) This setting controls whether left censored results are assumed to have a lower bound of 0 or whether the result can be negative.
- **precision_tolerance_to_drop_censor**: (default 0.25) This setting controls whether a range of possible results is returned as censored or non-censored. For example, if an average is known to be between 2 and 3, then 2.5 would be returned as the result since the whole range is within 20% of 2.5 (20% < 25%). If this parameter was set to 0.15 (15%), then the tolerance would not cover the range and a result of <3 or >2 would be returned depending on the value of focus_high_potential.
- **precision_rounding**: (default True) This setting controls whether to apply a specific rounding procedure.

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
import censoredsummarystats as censored

# Create example results
df = pd.DataFrame([['Set1',1.5],['Set1','2'],['Set1','>2.5'],['Set2',2],['Set2','<3'],['Set2',7.0]],columns=['Groups','Results'])

averages = censored.average(df,'Results',groupby_cols=['Groups'])
print(averages)
```
Output are like this:
```python
  Groups Statistic Result CensorComponent  NumericComponent      Interval
0   Set1   Average  >2.00               >               2.0   (2.00, inf)
1   Set2   Average   3.50                               3.5  [3.00, 4.00)
```