import numpy as np
import collections
from datetime import datetime


class DataType(object):
    """
    Conversion between numpy and python types for the Tree input data type.
    The upper data type for tree data.
    """

    def __init__(self, numpy_dtype, python_dtype, numpy_na_value, python_na_value):
        """
        Initialize the data type object.
        :param numpy_dtype: Specification of the numpy type
        :param python_dtype: Specification of the python type
        :param numpy_na_value: Specification of the numpy missing value
        :param python_na_value: Specification of the python missing value
        """
        self.numpy_dtype = numpy_dtype
        self.python_dtype = python_dtype
        self.numpy_na_value = numpy_na_value
        self.python_na_value = python_na_value

    def is_nullable(self):
        """
        Method returns whether the current data type is nullable.
        :return: Boolean
        """
        return self.python_na_value is not None or self.numpy_na_value is not None

    def get_numpy_type(self):
        """
        Method to return numpy type of the data type.
        :return: Numpy DType
        """
        return np.dtype(self.numpy_dtype)

    def get_python_type(self):
        """
        Method to return python type of the data type.
        :return: Type
        """
        return self.python_dtype

    def build_numpy_value(self, value):
        """
        Method which converts the input value into the numpy type.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        return self.get_numpy_type().type(value).astype(self.get_numpy_type())

    def build_python_value(self, value):
        """
        Nethod which converts the input value into the python type value.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        return self.get_python_type()(value)


class StringDataType(DataType):
    """
    DataType for string/categorical inputs.
    """

    def __init__(self, nullable=True, longest_string=200):
        """
        Initialize the data type.
        :param nullable: Boolean specifying whether the data type can contain missing values.
        :param longest_string: Integer specifying the longest possible string input.
        """
        if nullable:
            super(StringDataType, self).__init__('<U{}'.format(longest_string), str, 'nan', None)
        else:
            super(StringDataType, self).__init__('<U{}'.format(longest_string), str, None, None)


class FloatDataType(DataType):
    """
    DataType for float/continuous/discrete inputs.
    """

    def __init__(self, nullable=True, bits=8):
        """
        Initialize the data type.
        :param nullable: Boolean specifying whether the data type can contain missing values.
        :param bits: Integer specifying the number of bits to allocate in the memory for the float.
        """
        if nullable:
            super(FloatDataType, self).__init__('<f{}'.format(bits), float, np.nan, None)
        else:
            super(FloatDataType, self).__init__('<f{}'.format(bits), float, None, None)


class DateDataType(DataType):
    """
    DataType for date/timestamp inputs.
    """

    def __init__(self, nullable=True, resolution='s', format_string="%Y-%m-%d %H:%M:%S.%f"):
        """
        Initialize Date DataType.
        :param nullable: Boolean specifying whether the data type can contain missing values.
        :param resolution: String specifying the wanted numpy resolution of the date type.
        :param format_string: String Timestamp format.
        """
        if nullable:
            super(DateDataType, self).__init__('<M8[{}]'.format(resolution),
                                               lambda x: datetime.strptime(x, format_string),
                                               np.datetime64('NaT'), None)
        else:
            super(DateDataType, self).__init__('<M8[{}]'.format(resolution),
                                               lambda x: datetime.strptime(x, format_string),
                                               None, None)

    def build_numpy_value(self, value):
        """
        Method which converts the input value into the numpy type.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        return self.get_numpy_type().type(self.build_python_value(value)).astype(self.get_numpy_type())


class ArrayDataType(DataType):
    """
    DataType for arrays (lists of single type).
    """

    def __init__(self, element_data_type, nullable=True):
        """
        Initialize the data type.
        :param element_data_type: DataType specifying the data type of the array elements.
        :param nullable: Boolean specifying whether the data type can contain missing values.
        """
        if not isinstance(element_data_type, DataType):
            raise AttributeError("The array element has to be of DataType instance!")

        self.element_data_type = element_data_type
        self.element_numpy_type = element_data_type.get_numpy_type()

        if nullable:
            super(ArrayDataType, self).__init__(np.ndarray, list,
                                                np.array([], dtype=self.element_numpy_type), [])
        else:
            super(ArrayDataType, self).__init__(np.ndarray, list, None, None)

    def build_numpy_value(self, value):
        """
        Method which converts the input value into the numpy type.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        if not isinstance(value, (collections.Sequence, np.ndarray)) or isinstance(value, str):
            raise AttributeError("Incorrect format of input value!")

        built_value = [self.element_data_type.build_numpy_value(x) for x in value]
        return self.get_numpy_type().type(built_value).astype(self.element_numpy_type)

    def build_python_value(self, value):
        """
        Nethod which converts the input value into the python type value.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        if not isinstance(value, (collections.Sequence, np.ndarray)) or isinstance(value, str):
            raise AttributeError("Incorrect format of input value!")

        built_value = [self.element_data_type.build_python_value(x) for x in value]
        return self.get_python_type()(built_value)


class ListDataType(DataType):
    """
    DataType for lists/trees (list with elements of different data types)
    """

    def __init__(self, element_data_types, nullable=True):
        """
        Initialize the data type.
        :param element_data_types: List/Sequence of DataTypes
        :param nullable: Boolean specifying whether the data type can contain missing values.
        """
        if not isinstance(element_data_types, (collections.Sequence, np.ndarray)) or isinstance(element_data_types,
                                                                                                str):
            raise AttributeError("Incorrect format of input element data types!")

        for element in element_data_types:
            if not isinstance(element, DataType):
                raise AttributeError("Elements of the list have to be of DataType instance!")

        self.element_data_types = element_data_types
        self.element_numpy_types = self._get_numpy_dtypes()

        if nullable:
            super(ListDataType, self).__init__(np.ndarray, list, np.empty((0,), dtype=self.element_numpy_types), [])
        else:
            super(ListDataType, self).__init__(np.ndarray, list, None, None)

    def _get_numpy_dtypes(self):
        """
        Helper method to build input numpy dtypes for numpy structured array.
        :return: List of tuples of format (String of index, String of numpy DType)
        """
        return [('{}'.format(x), self.element_data_types[x].get_numpy_type()) for x in
                range(len(self.element_data_types))]

    def build_numpy_value(self, value):
        """
        Method which converts the input value into the numpy type.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        if not isinstance(value, (collections.Sequence, np.ndarray)) or isinstance(value, str):
            raise AttributeError("Incorrect format of input value!")

        input_values = [tuple([self.element_data_types[x].build_numpy_value(value[x])
                               for x in range(len(self.element_data_types))])]

        return np.array(input_values, dtype=self.element_numpy_types)

    def build_python_value(self, value):
        """
        Nethod which converts the input value into the python type value.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        if not isinstance(value, (collections.Sequence, np.ndarray)) or isinstance(value, str):
            raise AttributeError("Incorrect format of input value!")

        input_values = tuple([self.element_data_types[x].build_python_value(value[x])
                              for x in range(len(self.element_data_types))])

        return self.get_python_type()(input_values)
