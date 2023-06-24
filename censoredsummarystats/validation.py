'''
These functions validate user input values.

'''

import pandas as pd

#%% Validate CensoredData

def _validate_cdf(cdf):
    
    #%% Check data input
    
    # Check that supplied data is a DataFrame
    if not isinstance(cdf.data, pd.core.frame.DataFrame):
        raise ValueError('The data supplied to CensoredData() must be a '
            'pandas DataFrame. Instead, an object was passed with type: '
            f'{type(cdf.data).__name__}')
    
    # Ensure that provided data is a copy and not indexed
    cdf.data = cdf.data.copy().reset_index()
    
    #%% Check value_col exists as column in data
    
    if cdf.value_col not in cdf.data.columns:
        raise ValueError('The value column supplied to CensoredData() '
            'was not found as a column in the data. Columns include '
            f'{cdf.data.columns.to_list()} which does not include '
            f'{repr(cdf.value_col)}.')
    
    #%% Check the four stat settings
    
    boolean_settings = ['include_negative_interval',
                        'focus_high_potential',
                        'precision_rounding',
                        'thousands_comma',
                        'output_interval']
    
    for setting in boolean_settings:
        setting_mode = getattr(cdf, setting)
        if not isinstance(setting_mode, bool):
            raise ValueError(f'The value supplied to {setting} must be '
                'True or False. Instead, the value supplied was '
                f'{repr(setting_mode)}.')
    
    # True or False are considered integer types so check for boolean too
    if ((isinstance(cdf.precision_tolerance_to_drop_censor, bool)) |
        (not isinstance(cdf.precision_tolerance_to_drop_censor,
                       (int, float)))):
        raise ValueError(f'The value supplied to '
            'precision_tolerance_to_drop_censor must be numeric. Instead, '
            'the value supplied was '
            f'{repr(cdf.precision_tolerance_to_drop_censor)}.')
    
    #%% Check column names for conflicts
    
    # Check that columns created within methods don't conflict with column
    # names in provided data
    # Check that new column names are string values
    
    # Create list of column names that may be created
    created_columns = [attribute for attribute in cdf.__annotations__
                               if attribute.endswith('_col') and
                                   attribute != 'value_col']
    
    # Check the list for duplicates
    if len(created_columns) != len(set(created_columns)):
        raise ValueError('Two default column names have been set to the '
            'same name. Check that assigned values are unique and not '
            'identical to any default values.')
    
    
    for col in created_columns:
        col_name = getattr(cdf, col)
        if (col_name in cdf.data.columns) & (col_name != cdf.value_col):
            raise ValueError('The data contains a column named '
                f'"{col_name}" which may be used in the output. Either '
                'rename this column or provide an alternative value for '
                f'{col} when initialising CensoredDataFrame.')
        
        if not isinstance(col_name, str):
            raise ValueError('Text values are required for column names. '
                f'The column name {col_name} should be changed to a text '
                f'value for {col}.')
    
    # Check that the data has no columns starting with double underscore
    dunder_cols = [col for col in cdf.data if col.startswith('__')]
    if len(dunder_cols) != 0:
        raise ValueError('Columns starting with "__" are used within '
            'the stat methods. Rename the following columns: '
            f'{dunder_cols}')
    
    #%% Check that there are no nan values or empty strings
    if ((cdf.data[cdf.value_col].isnull().any()) |
        (cdf.data[cdf.value_col].astype(str).str.len().min() == 0)):
        raise ValueError('Missing values need to be removed from the data '
            'before it can be analysed.')

#%% Groupby validation

def _validate_groupby_cols(cdf, groupby_cols):
    
    # Skip None case
    if groupby_cols == None:
        pass
    else:
        # Check that groupby_cols is a nested list
        if ((not isinstance(groupby_cols, list)) | 
            (not (all(isinstance(group, list) for group in groupby_cols)))):
            raise ValueError('groupby_cols needs to be a nested list. '
                f'The provided input was {groupby_cols}')
        
        # Check that column names are included in data
        if not (all(name in cdf.data.columns for name in groupby_cols[0])):
            raise ValueError('Columns used in groupby_cols were not found in '
                'the DataFrame. Names not found include: '
                f'{[x for x in groupby_cols[0] if x not in cdf.data.columns]}')
        
        # Check that subsequent lists are subsets
        for i in range(len(groupby_cols)-1):
            # Create list of columns not in prior group
            errors = [x for x in groupby_cols[i+1] if x not in groupby_cols[i]]
            if len(errors) != 0:
                raise ValueError('When using tiered groupings, subsequent '
                    'groupings must be subsets of earlier groups. The '
                    'following column names were not included in all earlier '
                    f'groups: {errors}')
        
        # Check for null values in groupby columns
        if cdf.data[groupby_cols[0]].isnull().values.any():
            raise ValueError('Null values found in one of the columns used '
                'for grouping.')
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            