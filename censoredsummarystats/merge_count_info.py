'''
The function in this module support the ability to merge count information
to stat tables. Required information includes (i) a CensoredData input object
that contains the original data table, (ii) a DataFrame that contains the
final stat outputs, (iii) the specific column groupings used to combine data,
and (iv) the names to give the count for each grouping.

'''

def _merge_count_info(cdf, stat_data, groupby_cols, count_cols, filters):
    
    # Create a copy of the data
    df = cdf.data.copy()
    
    # Apply filters
    if filters:
        for col_name in filters:
            df = df[df[col_name].isin(filters[col_name])]
    
    # Validate count_cols
    _validate_count_cols(cdf, groupby_cols, count_cols)
    
    # If groupby_cols is None
    if groupby_cols == None:
        stat_data[count_cols[0]] = len(df)
    
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
                                    on=groupby_cols[-1],
                                    validate = '1:1')
    
    return stat_data


def _validate_count_cols(cdf, groupby_cols, count_cols):
    
    # Check that count_cols is a list
    if not isinstance(count_cols, list):
        raise ValueError('count_cols must be a list. '
            f'{count_cols=} was passed.')
    
    # Check that items are all strings
    if not (all(isinstance(name, str) for name in count_cols)):
        raise ValueError('count_cols must be a list of text strings. '
            f'Non-text names were detected in {count_cols=}')
    
    # Check that names are unique
    if (len(count_cols) != len(set(count_cols))):
        raise ValueError('count_cols must contain unique names. '
            f'Non-unique names were detected in {count_cols=}')
    
    # Check that single name provided if groupby_cols is None or []
    if groupby_cols == None:
        if len(count_cols) != 1:
            raise ValueError('A single column name should be provided as '
                'a list when groupby_cols is not used. '
                f'{count_cols=} was passed.')
    else:
        # Check that lists are same length if groupby_cols is used
        if len(count_cols) != len(groupby_cols):
            raise ValueError('count_cols must have the same number of '
                'column names as there are grouping lists in groupby_cols. '
                f'{len(count_cols)} names were provided in count_cols and '
                f'{len(groupby_cols)} names were provided in groupby_cols.')
        
        # Check names are not used in the groupby_cols
        groupby_names = sum(groupby_cols, [])
        common_names = [name for name in count_cols if name in groupby_names]
        if len(common_names) != 0:
            raise ValueError('count_cols cannot include names that already '
                'exist in the data which are also used for grouping. '
                f'Repeated names include: {common_names}')