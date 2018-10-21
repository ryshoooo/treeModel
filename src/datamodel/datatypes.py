import numpy as np
from datetime import date


class TreeDataType(object):
    """
    Conversion between numpy and python types for the Tree input data type.
    """

    def __init__(self, numpy_dtype, python_dtype, numpy_na_value, python_na_value):
        self.numpy_dtype = None
        self.python_dtype = None
        self.numpy_na_value = None
        self.python_na_value = None

    def is_nullable(self):
        return self.python_na_value is None or self.numpy_na_value


class StringDataType(TreeDataType):
    def __init__(self, nullable=True, longest_string=200):
        if nullable:
            super().__init__('<U{}'.format(longest_string), str, 'nan', None)
        else:
            super().__init__('<U{}'.format(longest_string), str, None, None)


class FloatDataType(TreeDataType):
    def __init__(self, nullable=True, bits=8):
        if nullable:
            super().__init__('<f{}'.format(bits), float, np.nan, None)
        else:
            super().__init__('<f{}'.format(bits), float, None, None)


class DateDataType(TreeDataType):
    def __init__(self, nullable=True, resolution='s'):
        if nullable:
            super().__init__('<M8[{}]'.format(resolution), date, np.datetime64('NaT'), None)
        else:
            super().__init__('<M8[{}]'.format(resolution), date, None, None)
