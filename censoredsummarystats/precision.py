'''
These functions apply a specific rounding algorithm based on value. Some values
are rounded based on specified decimal places while others are based on
significant digits.

'''

import numpy as np
from decimal import Decimal, ROUND_HALF_UP

def _string_precision(value,
                      thousands_comma=False):
    '''
    A function that applies a specified rounding method that is value
    dependent. This method is specifically designed to reflect the
    typical measurement precision for water quality results. Depending on the
    value, the rounding is either to a particular number of decimal places or
    to a particular number of significant digits.

    Parameters
    ----------
    value : float
        Numeric value that will be rounded.

    Returns
    -------
    string : string
        The rounded value expressed as a string to the appropriate precision.

    '''
    
    # Create string notation rounded to 6 significant digits
    string = f'{value:.6g}'
    
    # Calculate the absolute value
    abs_value = abs(value)
    
    # Calculate precision parameter based on value
    if round(abs_value,1) >= 100:
        precision = Decimal(f'{value:.3g}') # 3 sig figs
    elif round(abs_value,2) >= 10:
        precision = Decimal('0.1') # 1 decimal place
    elif round(abs_value,3) >= 0.2:
        precision = Decimal('0.01') # 2 decimal place
    elif round(abs_value,3) >= 0.1:
        precision = Decimal('0.001') # 3 decimal place
    else:
        precision = Decimal(f'{value:#.2g}') # 2 sig figs
    
    # Check for infinite values
    if abs_value == np.inf:
        pass
    else:
        # Apply appropriate precision
        string = str(Decimal(string).quantize(precision,
                                              rounding=ROUND_HALF_UP))
        # Values over 100 are returned without decimal places (as integers)
        if round(abs_value,1) >= 100:
            if thousands_comma:
                string = f'{int(float(string)):,}'
            else:
                string = f'{int(float(string))}'
        
    return string


def _numeric_precision(value):
    '''
    A function that returns a float data type from a rounding function instead
    of a string data type.

    Parameters
    ----------
    value : float
        Float value that may have more decimal places or significant digits
        than is appropriate

    Returns
    -------
    float
        Float value that is rounded appropriately

    '''
    
    # Return the same rounded result as string_precision but as a float
    return float(_string_precision(value))

