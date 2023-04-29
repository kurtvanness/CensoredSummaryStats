# Import numpy

import numpy as np

def string_precision(value):
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
    
    # Check for infinite values
    if abs(value) == np.inf:
        string = str(value)
    # Values above 100 or are rounded to 100 should be rounded to 3 significant digits
    # Furthermore, they should be expressed as integers using a comma as the thousands separator
    elif round(abs(value),1) >= 100:
        string = f'{value:.3g}'
        string = f'{int(float(string)):,}'
    # Values above 10 or are rounded to 10 should be rounded to 1 decimal place
    elif round(abs(value),2) >= 10:
        string = f'{value:.1f}'
    # Values above 0.2 or are rounded to 0.2 should be rounded to 2 decimal places
    elif round(abs(value),3) >= 0.2:
        string = f'{value:.2f}'
    # Values above 0.1 or are rounded to 0.1 should be rounded to 3 decimal places
    elif round(abs(value),3) >= 0.1:
        string = f'{value:.3f}'
    # Values below 0.1 should be rounded to 2 significant digits
    else:
        string = f'{value:.2g}'

    return string


def numeric_precision(value):
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
    return float(string_precision(value))

