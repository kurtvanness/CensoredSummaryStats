# censoredsummarystats
A repository that contains functions for calculating summary stats on censored data.

## Functions
Current functions include maximum, minimum, average, percentile, median (same as 50th percentile).

All functions require a dataframe that contains a column of results. This package is only useful when the results are written as strings that cannot be directly converted to a numeric datatype due to the presence of symbols that indicate the result is potentially above or below a particular value. The accepted censorship symbols include (<,≤,≥,>).

Additional table columns can be provided as a list so that the statistical functions obtain results for groups of results.

More options are available and described within each function.
