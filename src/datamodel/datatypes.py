import numpy as np
import collections
from datetime import date


class DataType(object):
    """
    Conversion between numpy and python types for the Tree input data type.
    """

    def __init__(self, numpy_dtype, python_dtype, numpy_na_value, python_na_value):
        self.numpy_dtype = numpy_dtype
        self.python_dtype = python_dtype
        self.numpy_na_value = numpy_na_value
        self.python_na_value = python_na_value

    def is_nullable(self):
        return self.python_na_value is not None or self.numpy_na_value is not None

    def get_numpy_type(self):
        return np.dtype(self.numpy_dtype)

    def get_python_type(self):
        return self.python_dtype

    def build_numpy_value(self, value):
        return self.get_numpy_type().type(value)

    def build_python_value(self, value):
        return self.get_python_type()(value)


class StringDataType(DataType):
    def __init__(self, nullable=True, longest_string=200):
        if nullable:
            super(StringDataType, self).__init__('<U{}'.format(longest_string), str, 'nan', None)
        else:
            super(StringDataType, self).__init__('<U{}'.format(longest_string), str, None, None)


class FloatDataType(DataType):
    def __init__(self, nullable=True, bits=8):
        if nullable:
            super(FloatDataType, self).__init__('<f{}'.format(bits), float, np.nan, None)
        else:
            super(FloatDataType, self).__init__('<f{}'.format(bits), float, None, None)


class DateDataType(DataType):
    def __init__(self, nullable=True, resolution='s'):
        if nullable:
            super(DateDataType, self).__init__('<M8[{}]'.format(resolution), date, np.datetime64('NaT'), None)
        else:
            super(DateDataType, self).__init__('<M8[{}]'.format(resolution), date, None, None)


class ArrayDataType(DataType):
    def __init__(self, element_data_type, nullable=True):
        if not isinstance(element_data_type, DataType):
            raise AttributeError("The array element has to be of DataType instance!")

        self.element_data_type = element_data_type

        if nullable:
            super(ArrayDataType, self).__init__(np.ndarray, list,
                                                np.array([], dtype=np.dtype(element_data_type.numpy_dtype)), [])
        else:
            super(ArrayDataType, self).__init__(np.ndarray, list, None, None)

    def build_numpy_value(self, value):
        if not isinstance(value, (collections.Sequence, np.ndarray)) or isinstance(value, str):
            raise AttributeError("Incorrect format of input value!")

        return super(ArrayDataType, self).build_numpy_value(value).astype(self.element_data_type.get_numpy_type())


class ListDataType(DataType):
    def __init__(self, element_data_types, nullable=True):
        if not isinstance(element_data_types, (collections.Sequence, np.ndarray)) or isinstance(element_data_types,
                                                                                                str):
            raise AttributeError("Incorrect format of input element data types!")

        for element in element_data_types:
            if not isinstance(element, DataType):
                raise AttributeError("Elements of the list have to be of DataType instance!")

        self.element_data_types = element_data_types

        if nullable:
            super(ListDataType, self).__init__(np.ndarray, list, np.empty((0,),
                                                                          dtype=self._get_numpy_dtypes()), [])
        else:
            super(ListDataType, self).__init__(np.ndarray, list, None, None)

    def _get_numpy_dtypes(self):
        return [('{}'.format(x), self.element_data_types[x].get_numpy_type()) for x in
                range(len(self.element_data_types))]

    def build_numpy_value(self, value):
        if not isinstance(value, (collections.Sequence, np.ndarray)) or isinstance(value, str):
            raise AttributeError("Incorrect format of input value!")

        input_values = tuple([self.element_data_types[x].build_numpy_value(value[x])
                              for x in range(len(self.element_data_types))])

        return np.array(input_values, dtype=self._get_numpy_dtypes())
