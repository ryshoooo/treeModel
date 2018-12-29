"""
This module contains implementation of standard data types used for creating and setting
tree schemas for input tree data.
"""
import numpy as np
import collections
from datetime import datetime
from copy import deepcopy
from warnings import warn


#####################################################
#              STANDARD DATATYPES                   #
#####################################################

class DataType(object):
    """
    Base DataType class. Contains data type necessary functionality to work in tree schemas.
    The subclasses of the ``DataType`` including the class itself are comparable, i.e. relations ``'<='``, ``'>='`` etc.
    are working. The meaning of comparisons is to efficiently determine subdata types and supdata types.

    Each ``DataType`` class contains the logic of value conversions from the input data to the specified type.
    Each ``DataType`` contains the conversion logic for both ``'numpy'`` and ``'python'`` methods.

    :param numpy_dtype: Specification of the numpy type.
    :param python_dtype: Specification of the python type.
    :param numpy_na_value: Specification of the numpy missing value.
    :param python_na_value: Specification of the python missing value.

    :type numpy_dtype: str
    :type python_dtype: str
    :type numpy_na_value: Singleton
    :type python_na_value: Singleton

    :ivar numpy_dtype: Specification of the numpy type.
    :ivar python_dtype: Specification of the python type.
    :ivar numpy_na_value: Specification of the numpy missing value.
    :ivar python_na_value: Specification of the python missing value.

    :vartype numpy_dtype: str
    :vartype python_dtype: str
    :vartype numpy_na_value: Singleton
    :vartype python_na_value: Singleton
    """

    def __init__(self, numpy_dtype, python_dtype, numpy_na_value, python_na_value):
        """
        Initialize the data type object.
        """
        self.numpy_dtype = numpy_dtype
        self.python_dtype = python_dtype
        self.numpy_na_value = numpy_na_value
        self.python_na_value = python_na_value

    def is_nullable(self):
        """
        Method returns whether the current data type is nullable, i.e. whether missing values are allowed.

        :return: ``True`` or ``False`` specifying whether missing values are allowed for the data type.
        :rtype: bool
        """
        return self.python_na_value is not None or self.numpy_na_value is not None

    def get_numpy_type(self):
        """
        Builds numpy type to be used for conversion.

        :return: Numpy data type which is to be applied for any input value.
        :rtype: :class:`np.dtype`
        """
        return np.dtype(self.numpy_dtype)

    def get_python_type(self):
        """
        Builds python type to be used for conversion.

        :return: Python data type which is to be applied for any input value.
        :rtype: type
        """
        return self.python_dtype

    def build_numpy_value(self, value):
        """
        Conversion method, which transforms input value into the numpy-typed value.

        :param value: Input datum.
        :type value: any

        :return: Converted value from the input datum to the specific data type.
        :rtype: Type from the :meth:`get_numpy_type` method.
        """
        return self.get_numpy_type().type(value).astype(self.get_numpy_type())

    def build_python_value(self, value):
        """
        Conversion method, which transforms input value into the python-typed value.

        :param value: Input datum.
        :type value: any

        :return: Converted value from the input datum to the specific data type.
        :rtype: Type from the :meth:`get_python_type` method.
        """
        try:
            return self.get_python_type()(value)
        except TypeError:
            return self.python_na_value

    def _compare(self, other, method):
        """
        Generic method to compare data type to other data type.
        :param other: DataType
        :param method: String
        :return: Boolean
        """
        if not isinstance(other, DataType):
            raise AttributeError("Cannot compare DataType to '{}'".format(type(other)))

        if isinstance(other, StringDataType) and method in ('__le__', '__lt__'):
            return True
        elif method in ('__le__', '__ge__'):
            return self == other
        else:
            return False

    def __str__(self):
        return "DataType"

    def __eq__(self, other):
        if not isinstance(other, DataType):
            raise AttributeError("Cannot compare DataType to '{}'".format(type(other)))
        else:
            return self.numpy_dtype == other.numpy_dtype and str(self.numpy_na_value) == str(other.numpy_na_value) and \
                   self.python_dtype == other.python_dtype and self.python_na_value == other.python_na_value

    def __le__(self, other):
        return self._compare(other, self.__le__.__name__)

    def __lt__(self, other):
        return self._compare(other, self.__lt__.__name__)

    def __ge__(self, other):
        return self._compare(other, self.__ge__.__name__)

    def __gt__(self, other):
        return self._compare(other, self.__gt__.__name__)


class StringDataType(DataType):
    """
    DataType for string/categorical inputs.

    NB: The max length for the strings is currently statically set to ``'128'`` characters only, in case ``'numpy'`` implementation
    of the StringDataType is being used. In case of ``'python'`` implementation, all string lengths are stored.

    :param nullable: ``True`` or ``False`` specifying whether the data type can contain missing values.
    :type nullable: bool

    :ivar numpy_na_value: ``'nan'``
    :ivar python_na_value: ``None``
    """

    def __init__(self, nullable=True):
        """
        Initialize the data type.
        """
        if nullable:
            super(StringDataType, self).__init__('<U128', str, 'nan', None)
        else:
            super(StringDataType, self).__init__('<U128', str, None, None)

    def _compare(self, other, method):
        if not isinstance(other, DataType):
            raise AttributeError("Cannot compare StringDataType to '{}'".format(type(other)))

        if method == '__ge__':
            return True
        elif method == '__lt__':
            return False
        elif method == '__gt__':
            return not self == other
        else:
            return self == other

    def __str__(self):
        return "StringDataType"

    def __eq__(self, other):
        if not isinstance(other, DataType):
            raise AttributeError("Cannot compare StringDataType to '{}'".format(type(other)))
        elif not isinstance(other, StringDataType):
            warn(message="StringDataType is not a {}".format(type(other)), category=UserWarning)
            return False
        else:
            return True


class FloatDataType(DataType):
    """
    DataType for float/continuous/discrete inputs.

    NB: The current numpy implementation of floats is using numpy data type ``'<f8'``.

    :param nullable: ``True`` or ``False`` specifying whether the data type can contain missing values.
    :type nullable: bool

    :ivar bits: Number of bits to allocate in the memory for the numpy data type.
    :vartype bits: int

    :ivar numpy_na_value: ``np.nan``
    :ivar python_na_value: ``None``
    """

    def __init__(self, nullable=True):
        """
        Initialize the data type.
        """
        self.bits = 8

        if nullable:
            super(FloatDataType, self).__init__('<f{}'.format(self.bits), float, np.nan, None)
        else:
            super(FloatDataType, self).__init__('<f{}'.format(self.bits), float, None, None)

    def _compare(self, other, method):
        if not isinstance(other, DataType):
            raise AttributeError("Cannot compare FloatDataType to '{}'".format(type(other)))

        if method == '__le__':
            return isinstance(other, (StringDataType, FloatDataType))
        elif method == '__lt__':
            return isinstance(other, StringDataType)
        elif method == '__ge__':
            return self == other
        else:
            return False

    def __str__(self):
        return "FloatDataType"

    def __eq__(self, other):
        if not isinstance(other, DataType):
            raise AttributeError("Cannot compare FloatDataType to '{}'".format(type(other)))
        elif not isinstance(other, FloatDataType):
            warn("FloatDataType is not a {}".format(type(other)), UserWarning)
            return False
        else:
            return True


class DateDataType(DataType):
    """
    DataType for date/timestamp inputs.
    The expected input data are to be strings, the conversion from string to the datetime types happens within
    the implementation of this class.

    :param nullable: ``True`` or ``False`` specifying whether the data type can contain missing values.
    :param resolution: Single value specifying the wanted numpy resolution of the date type (e.g. ``'s'``, ``'h'``, ``'M'``, etc.).
    :param format_string: The input format of the timestamp. The formatting follows the standards specified in the :mod:`datetime` documentation. TODO: LINK FOR FORMATS

    :type nullable: bool
    :type resolution: str
    :type format_string: str

    :ivar resolution: Single value specifying the wanted numpy resolution of the date type (e.g. ``'s'``, ``'h'``, ``'M'``, etc.).
    :ivar format_string: The input format of the timestamp. The formatting follows the standards specified in the :mod:`datetime` documentation. TODO: LINK FOR FORMATS
    :vartype resolution: str
    :vartype format_string: str

    :ivar numpy_na_value: ``np.NaT``
    :ivar python_na_value: ``None``
    """

    def __init__(self, nullable=True, resolution='s', format_string="%Y-%m-%d %H:%M:%S.%f"):
        """
        Initialize Date DataType.
        """
        self.resolution = resolution
        self.format_string = format_string
        if nullable:
            super(DateDataType, self).__init__('<M8[{}]'.format(resolution), self._datetime_format, '', '')
        else:
            super(DateDataType, self).__init__('<M8[{}]'.format(resolution), self._datetime_format, None, None)

    def _datetime_format(self, value):
        """
        Helper method to convert input value into the python datetime format.
        :param value: String representing the timestamp
        :param format_string: String representing the timestamp format
        :return: Either datetime object or empty string.
        """
        try:
            return datetime.strptime(value, self.format_string)
        except ValueError:
            return ''

    def build_numpy_value(self, value):
        """
        Overrides the :meth:`DataType.build_numpy_value` method from the super class. It is necessary to override the
        base method, since currently numpy datetime64 module contains different datetime formatting as datetime module.
        For formatting unification, only ``datetime`` implementation is being used, thus has to be enforced in this
        method as well.

        :param value: Value to be converted.
        :type value: str
        :return: Converted datetime value from the input value.
        :rtype: :class:`np.datetime64`
        """
        return self.get_numpy_type().type(self.build_python_value(value)).astype(self.get_numpy_type())

    def _compare(self, other, method):
        """
        Generic method to compare DateDataType to a DataType.
        :param other: DataType
        :param method: String
        :return: Boolean
        """
        if not isinstance(other, DataType):
            raise AttributeError("Cannot compare DateDataType to '{}'".format(type(other)))

        if isinstance(other, StringDataType):
            return method in ('__lt__', '__le__')
        elif isinstance(other, DateDataType):
            return self.get_numpy_type().__getattribute__(method)(other.get_numpy_type())

    def __str__(self):
        return "DateDataType({}, {})".format(self.resolution, self.format_string)

    def __eq__(self, other):
        if not isinstance(other, DataType):
            raise AttributeError("Cannot compare DateDataType to '{}'".format(type(other)))
        elif not isinstance(other, DateDataType):
            warn("DateDataType is not a {}".format(type(other)), UserWarning)
            return False
        else:
            return self.resolution == other.resolution


class ArrayDataType(DataType):
    """
    DataType for arrays (lists of single type).

    :param element_data_type: Specifies the singular data type of the array elements.
    :param nullable: ``True`` or ``False`` specifying whether the data type can contain missing values.

    :type element_data_type: DataType
    :type nullable: bool

    :ivar element_data_type: Specifies the singular data type of the array elements.
    :ivar element_numpy_type: Built numpy type from the :meth:`DataType.get_numpy_type` method of the ``element_data_type`` parameter.

    :vartype element_data_type: DataType
    :vartype element_numpy_type: :class:`np.dtype`

    :ivar numpy_na_value: empty numpy array
    :ivar python_na_value: empty list
    """

    def __init__(self, element_data_type, nullable=True):
        """
        Initialize the data type.
        """
        if not isinstance(element_data_type, DataType):
            raise AttributeError("The array element has to be of DataType instance!")

        self.element_data_type = deepcopy(element_data_type)
        self.element_numpy_type = self.element_data_type.get_numpy_type()

        if nullable:
            super(ArrayDataType, self).__init__(np.ndarray, list,
                                                np.array([], dtype=self.element_numpy_type), [])
        else:
            super(ArrayDataType, self).__init__(np.ndarray, list, None, None)

    def build_numpy_value(self, value):
        """
        This method applies the built numpy type in attribute ``element_numpy_type`` to each value in the input list.
        Finally converts the whole list into a numpy array.

        :param value: Value to be converted.
        :type value: :class:`np.ndarray` or list

        :return: Converted value of the specific data type.
        :rtype: :class:`np.ndarray`
        """
        if not isinstance(value, (collections.Sequence, np.ndarray)) or isinstance(value, str):
            raise AttributeError("Incorrect format of input value!")

        built_value = [self.element_data_type.build_numpy_value(x) for x in value]
        return self.get_numpy_type().type(built_value).astype(self.element_numpy_type)

    def build_python_value(self, value):
        """
        Converts each element of the input list to the specified python type of the ``element_data_type``.
        Finally returns as a python list of converted values.

        :param value: Value to be converted.
        :type value: list(any)

        :return: Converted value of the specific data type.
        :rtype: list(:meth:`DataType.get_python_type`)
        """
        if not isinstance(value, (collections.Sequence, np.ndarray)) or isinstance(value, str):
            raise AttributeError("Incorrect format of input value!")

        built_value = [self.element_data_type.build_python_value(x) for x in value]
        return self.get_python_type()(built_value)

    def _compare(self, other, method):
        """
        Generic method to compare Array Data Type to other Data Type.
        :param other: DataType
        :param method: String
        :return: Boolean
        """
        if not isinstance(other, DataType):
            raise AttributeError("Cannot compare ArrayDataType to '{}'".format(type(other)))

        if isinstance(other, (StringDataType, ListDataType)):
            return method in ('__le__', '__lt__')
        elif isinstance(other, ArrayDataType):
            return self.element_data_type.__getattribute__(method)(other.element_data_type)
        else:
            return False

    def __str__(self):
        return """ArrayDataType({})""".format(self.element_data_type)

    def __eq__(self, other):
        if not isinstance(other, DataType):
            raise AttributeError("Cannot compare ArrayDataType to '{}'".format(type(other)))
        elif not isinstance(other, ArrayDataType):
            warn("ArrayDataType is not {}".format(type(other)), UserWarning)
            return False
        else:
            return self.element_data_type == other.element_data_type


class ListDataType(DataType):
    """
    DataType for lists (list with elements of different data types)

    :param element_data_types: Specifies the data types of the list elements in the input order.
    :param nullable: ``True`` or ``False`` specifying whether the data type can contain missing values.
    :param level: Specifies the positioning of the data type in the tree schema from the root of the tree (starts with 1). This value is needed for the hidden string method :meth:`ListDataType.__str__`, which correctly formats the input data types based on the level information.

    :type element_data_types: list(DataType)
    :type nullable: bool
    :type level: int

    :ivar element_data_types: Specifies the singular data type of the array elements.
    :ivar element_numpy_types: Built numpy types from the :meth:`DataType.get_numpy_type` prepared as an input value for :class:`np.ndarray` ``dtype`` parameter of the structured numpy array.
    :ivar level: Specifies the positioning of the data type in the tree schema.

    :vartype element_data_types: list(DataType)
    :vartype element_numpy_types: (str, :class:`np.dtype`)
    :vartype level: int

    :ivar numpy_na_value: empty numpy structured array
    :ivar python_na_value: empty list
    """

    def __init__(self, element_data_types, nullable=True, level=1):
        """
        Initialize the data type.
        """
        if not isinstance(element_data_types, (collections.Sequence, np.ndarray)) or isinstance(element_data_types,
                                                                                                str):
            raise AttributeError("Incorrect format of input element data types!")

        for element in element_data_types:
            if not isinstance(element, DataType):
                raise AttributeError("Elements of the list have to be of DataType instance!")

        self.element_data_types = deepcopy(element_data_types)
        self.element_numpy_types = self._get_numpy_dtypes()
        self.level = level

        if nullable:
            super(ListDataType, self).__init__(np.ndarray, list, np.empty((0,), dtype=self.element_numpy_types), [])
        else:
            super(ListDataType, self).__init__(np.ndarray, list, None, None)

    def _get_numpy_dtypes(self):
        """
        Helper method to build input numpy dtypes for numpy structured array.
        :return: List of tuples of format (String of index, String of numpy DType)
        """
        return [(str(_ind), _dt.get_numpy_type()) for _ind, _dt in enumerate(self.element_data_types)]

    def build_numpy_value(self, value):
        """
        This method converts each input value to its corresponding numpy data type and finally converts the collection
        into the numpy structured array.

        :param value: Value to be converted.
        :type value: :class:`np.ndarray` or list

        :return: Converted value of the specific data type.
        :rtype: :class:`np.ndarray`
        """
        if not isinstance(value, (collections.Sequence, np.ndarray)) or isinstance(value, str):
            raise AttributeError("Incorrect format of input value!")

        if isinstance(value, np.ndarray) and value.dtype.type == np.void:
            value = list(value[0])

        built_np_vals = [_dt.build_numpy_value(value[_ind]) for _ind, _dt in enumerate(self.element_data_types)]
        input_values = [tuple(built_np_vals)]

        return np.array(input_values, dtype=self.element_numpy_types)

    def build_python_value(self, value):
        """
        Converts each element of the input list to its corresponding python data type.
        Finally returns as a python list of converted values.

        :param value: Value to be converted.
        :type value: list(any)

        :return: Converted value of the specific data type.
        :rtype: list
        """
        if not isinstance(value, (collections.Sequence, np.ndarray)) or isinstance(value, str):
            raise AttributeError("Incorrect format of input value!")

        input_values = tuple([_dt.build_python_value(value[_ind]) for _ind, _dt in enumerate(self.element_data_types)])

        return self.get_python_type()(input_values)

    def _compare(self, other, method):
        if not isinstance(other, DataType):
            raise AttributeError("Cannot compare ListDataType to '{}'".format(type(other)))

        if isinstance(other, StringDataType):
            return method in ('__le__', '__lt__')
        elif isinstance(other, ArrayDataType):
            return method in ('__ge__', '__gt__')
        elif isinstance(other, ListDataType):
            return all([data_type.__getattribute__(method)(other.element_data_types[ind])
                        for ind, data_type in enumerate(self.element_data_types)])
        else:
            return False

    def __str__(self):
        """
        Format list data type into a string, add tab for each element based on the current level.
        :return: String
        """
        elements_str = ("\n" + "\t" * self.level).join([str(x) for x in self.element_data_types])

        return str("ListDataType(\n" + "\t" * self.level + "{}\n" + "\t" * (self.level - 1) + " " * len(
            "ListDataType") + ")").format(elements_str)

    def __eq__(self, other):
        if not isinstance(other, DataType):
            raise AttributeError("Cannot compare ListDataType to '{}'".format(type(other)))
        elif not isinstance(other, ListDataType):
            warn("ListDataType is not a {}".format(type(other)), UserWarning)
            return False
        elif len(self.element_data_types) != len(other.element_data_types):
            warn("Non-equal lengths of lists!", UserWarning)
            return False
        else:
            return all([x[0] == x[1] for x in zip(self.element_data_types, other.element_data_types)])
