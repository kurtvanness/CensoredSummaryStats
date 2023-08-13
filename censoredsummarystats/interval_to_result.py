'''
The functions in this module support the ability to convert bound and
boundary information into a single result. Required information includes
(i) a CensoredData object that contains the column names and parameter
settings, and (ii) a DataFrame that contains boundary information.

'''

import numpy as np
from censoredsummarystats.precision import _string_precision, _numeric_precision

def _interval_notation(cdf, stat_data):
    
    # Create a copy of the data
    df = stat_data.copy()
    
    # Determine the left boundary symbol
    df[cdf.interval_col] = (
        np.where(df[cdf.left_boundary_col] == 'Open','(','['))
    
    # Incorporate left and right bounds
    df[cdf.interval_col] += (
        df[cdf.left_bound_col].astype(str) + ', ' + 
        df[cdf.right_bound_col].astype(str)
        )
    
    # Determine the right boundary symbol
    df[cdf.interval_col] += (
        np.where(df[cdf.right_boundary_col] == 'Open',')',']'))
    
    return df

def _components_from_interval(cdf, stat_data, focus_high_potential):
    
    # Create a copy of the data
    df = stat_data.copy()
    
    # Determine the midpoint for the interval if finite interval
    df[cdf._midpoint_col] = (
        np.where((df[cdf.left_bound_col] > -np.inf) & 
                 (df[cdf.right_bound_col] < np.inf),
            0.5 * (df[cdf.left_bound_col] + df[cdf.right_bound_col]),
            np.nan
            )
        )
    
    # Start with the conditions that produce an uncensored result
    conditions = [
        # If the bounds are equal, then the result is uncensored
        (df[cdf.left_bound_col] == df[cdf.right_bound_col]) |
        # If the bounds are finite and the interval is within the 
        # precision tolerance, then the result is uncensored
        ((df[cdf.right_bound_col] - df[cdf._midpoint_col]) <= 
            df[cdf._midpoint_col] * cdf.precision_tolerance_to_drop_censor)
        ]
    
    censor_results = [
        ''
        ]
    
    numeric_results =[
        df[cdf._midpoint_col]
        ]
    
    # If focused on the highest potential result
    if focus_high_potential:
        # Set censor and numeric components for each condition
        conditions += [
            # If there is an infinite right bound,
            # then the result is right censored
            # where closed boundary indicates potential equality
            (
                (df[cdf.right_bound_col] == np.inf) & 
                (df[cdf.left_bound_col] > -np.inf) & 
                (df[cdf.left_boundary_col] == 'Open')
            ),
            (
                (df[cdf.right_bound_col] == np.inf) & 
                (df[cdf.left_bound_col] > -np.inf) & 
                (df[cdf.left_boundary_col] == 'Closed')
            ),
            # Otherwise, the result is left censored
            # where closed boundary indicates potential equality
            (
                (df[cdf.right_bound_col] < np.inf) & 
                (df[cdf.right_boundary_col] == 'Open')
            ),
            (
                (df[cdf.right_bound_col] < np.inf) & 
                (df[cdf.right_boundary_col] == 'Closed')
            )
            ]
        
        censor_results += [
            '>',
            '≥',
            '<',
            '≤'
            ]
        
        numeric_results += [
            df[cdf.left_bound_col],
            df[cdf.left_bound_col],
            df[cdf.right_bound_col],
            df[cdf.right_bound_col]
            ]
    # Else focused on the lowest potential result
    else:
        # Determine the lower bound
        if cdf.include_negative_interval:
            lower_bound = -np.inf
        else:
            lower_bound = 0.0
        # Set censor and numeric components for each condition
        conditions += [
            # If the left bound is identical to the lower bound,
            # then the result is left censored
            # where closed boundary indicates potential equality
            (
                (df[cdf.left_bound_col] == lower_bound) & 
                (df[cdf.right_bound_col] < np.inf) & 
                (df[cdf.right_boundary_col] == 'Open')
            ),
            (
                (df[cdf.left_bound_col] == lower_bound) & 
                (df[cdf.right_bound_col] < np.inf) & 
                (df[cdf.right_boundary_col] == 'Closed')
            ),
            # Otherwise, the result is right censored
            # where closed boundary indicates potential equality
            (
                (df[cdf.left_bound_col] > -np.inf) &
                (df[cdf.left_boundary_col] == 'Open')
            ),
            (
                (df[cdf.left_bound_col] > -np.inf) &
                (df[cdf.left_boundary_col] == 'Closed')
            )
            ]
        
        censor_results += [
            '<',
            '≤',
            '>',
            '≥'
            ]
        
        numeric_results += [
            df[cdf.right_bound_col],
            df[cdf.right_bound_col],
            df[cdf.left_bound_col],
            df[cdf.left_bound_col]
            ]
    
    # Determine the censor and numeric components
    # If no condition is met, default to <> and NaN
    df[cdf.censor_col] = np.select(conditions, censor_results, '<>')
    df[cdf.numeric_col] = np.select(conditions, numeric_results, np.nan)
    if cdf.precision_rounding:
        df[cdf.numeric_col] = df[cdf.numeric_col].apply(_numeric_precision)
    
    # Drop working columns
    df = df.drop([cdf._midpoint_col], axis=1)
    
    return df
    
def _result_from_components(cdf, stat_data):
    
    # Create a copy of the data
    df = stat_data.copy()
    
    # Combine the censor and numeric components to create a combined result
    # Apply the appropriate rounding functions if precision_rounding = True
    if cdf.precision_rounding:
        df[cdf.result_col] = (
            df[cdf.censor_col] + 
            df[cdf.numeric_col].apply(_string_precision, cdf.thousands_comma)
            )
    else:
        df[cdf.result_col] = (
            df[cdf.censor_col] + 
            df[cdf.numeric_col].astype(str)
            )
    
    return df

def _interval_to_result(cdf, stat_data, focus_high_potential=None):
    
    # Create a copy of the data
    df = stat_data.copy()
    
    # Create column with interval notation
    if cdf.output_interval:
        df = _interval_notation(cdf, df)
    
    # Use cdf setting if notsert explicitly
    if focus_high_potential == None:
        focus_high_potential = cdf.focus_high_potential
    
    # Convert the interval for the stat into censor and numeric notation
    df = _components_from_interval(cdf, df, focus_high_potential)
    
    # Combine the censor and numeric components into a result
    df = _result_from_components(cdf, df)
    
    # Drop columns
    df = df.drop(columns=[cdf.left_bound_col,
                          cdf.left_boundary_col,
                          cdf.right_bound_col,
                          cdf.right_boundary_col])
    
    # Reorder columns
    df[cdf.censor_col] = df.pop(cdf.censor_col)
    df[cdf.numeric_col] = df.pop(cdf.numeric_col)
    if cdf.output_interval:
        df[cdf.interval_col] = df.pop(cdf.interval_col)
    
    return df
