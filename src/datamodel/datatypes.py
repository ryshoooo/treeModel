import numpy as np
import collections
from datetime import datetime


#####################################################
#              STANDARD DATATYPES                   #
#####################################################

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
        Method which converts the input value into the python type value.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        try:
            return self.get_python_type()(value)
        except TypeError:
            return self.python_na_value

    def __str__(self):
        return "DataType"


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
        self.longest_string = longest_string

        if nullable:
            super(StringDataType, self).__init__('<U{}'.format(longest_string), str, 'nan', None)
        else:
            super(StringDataType, self).__init__('<U{}'.format(longest_string), str, None, None)

    def __str__(self):
        return "StringDataType"


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
        self.bits = bits
        if nullable:
            super(FloatDataType, self).__init__('<f{}'.format(bits), float, np.nan, None)
        else:
            super(FloatDataType, self).__init__('<f{}'.format(bits), float, None, None)

    def __str__(self):
        return "FloatDataType"


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
        self.resolution = resolution
        self.format_string = format_string
        if nullable:
            super(DateDataType, self).__init__('<M8[{}]'.format(resolution),
                                               lambda x: self._datetime_format(x, format_string),
                                               '', '')
        else:
            super(DateDataType, self).__init__('<M8[{}]'.format(resolution),
                                               lambda x: self._datetime_format(x, format_string),
                                               None, None)

    @staticmethod
    def _datetime_format(value, format_string):
        """
        Helper method to convert input value into the python datetime format.
        :param value: String representing the timestamp
        :param format_string: String representing the timestamp format
        :return: Either datetime object or empty string.
        """
        try:
            return datetime.strptime(value, format_string)
        except ValueError:
            return ''

    def build_numpy_value(self, value):
        """
        Method which converts the input value into the numpy type.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        return self.get_numpy_type().type(self.build_python_value(value)).astype(self.get_numpy_type())

    def __str__(self):
        return "DateDataType({}, {})".format(self.resolution, self.format_string)


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
        Method which converts the input value into the python type value.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        if not isinstance(value, (collections.Sequence, np.ndarray)) or isinstance(value, str):
            raise AttributeError("Incorrect format of input value!")

        built_value = [self.element_data_type.build_python_value(x) for x in value]
        return self.get_python_type()(built_value)

    def __str__(self):
        return """ArrayDataType({})""".format(self.element_data_type)


class ListDataType(DataType):
    """
    DataType for lists (list with elements of different data types)
    """

    def __init__(self, element_data_types, nullable=True, level=1):
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
        Method which converts the input value into the python type value.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        if not isinstance(value, (collections.Sequence, np.ndarray)) or isinstance(value, str):
            raise AttributeError("Incorrect format of input value!")

        input_values = tuple([self.element_data_types[x].build_python_value(value[x])
                              for x in range(len(self.element_data_types))])

        return self.get_python_type()(input_values)

    def __str__(self):
        return str(
            "ListDataType(\n" + "\t" * self.level + "{}\n" + "\t" * (self.level - 1) +
            " " * len("ListDataType") + ")").format(
            ("\n" + "\t" * self.level).join([str(x) for x in self.element_data_types]))


#####################################################
#              TREE FUNCTIONALITY                   #
#####################################################

class TreeDataType(DataType):
    """
    DataType for trees (python dictionaries).
    """

    def __init__(self, schema, nullable=True):
        """
        Initialize the data type.
        :param schema: TreeSchema specifying the input tree.
        :param nullable: Boolean specifying whether the data type can contain missing values.
        """

        if not isinstance(schema, TreeSchema):
            raise AttributeError("Input schema has to be an instance of TreeSchema!")

        self.schema = schema

        if nullable:
            super(TreeDataType, self).__init__(dict, dict, {}, {})
        else:
            super(TreeDataType, self).__init__(dict, dict, None, None)

    def build_numpy_value(self, value):
        """
        Method which converts the input value into the numpy type.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        if not isinstance(value, dict):
            raise AttributeError("Cannot build non-dictionary-like input in TreeDataType!")

        return self.schema.base_fork_node.build_value(self.get_numpy_type().type(value), 'numpy')

    def build_python_value(self, value):
        """
        Method which converts the input value into the python type value.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        if not isinstance(value, dict):
            raise AttributeError("Cannot build non-dictionary-like input in TreeDataType!")

        return self.schema.base_fork_node.build_value(self.get_python_type()(value), 'python')

    def __str__(self):
        return """TreeDataType({})""".format(str(self.schema))


class Node(object):
    """
    Main node object, can be considered as a data point or a collection of data points.
    """

    def __init__(self):
        """
        Instantiation of the main node object.
        """
        # Set children nodes, name, value and the data_type of the node to None.
        self.children = None
        self.name = None
        self.data_type = None
        self.value = None

    def is_child(self):
        """
        Simple method to determine whether the node is a leaf.
        :return: Boolean
        """
        return self.children is None and self.name is not None and self.data_type is not None and not isinstance(
            self.data_type, TreeDataType)

    def is_fork(self):
        """
        Simple method to determine whether the node is forking.
        :return: Boolean
        """
        return self.children is not None and self.name is not None and self.data_type is not None and isinstance(
            self.data_type, TreeDataType)

    def get_name(self):
        """
        Get the name of the node.
        :return: String.
        """
        if self.name is None:
            raise RuntimeError("The name of the node is missing!")

        return self.name

    def set_name(self, name):
        """
        Set the name of the node.
        :param name: String
        :return: Node with name set.
        """
        if not isinstance(name, str):
            raise AttributeError("Parameter name has to be a string!")
        self.name = name

        return self

    def get_data_type(self):
        """
        Get the DataType of the node.
        :return: DataType.
        """
        if self.data_type is None:
            raise RuntimeError("The data type is missing!")

        return self.data_type

    def set_data_type(self, data_type):
        """
        Set the data type of the node.
        :param data_type: DataType object.
        :return: Instance of self with updated data type.
        """
        if not isinstance(data_type, DataType):
            raise AttributeError("Parameter data_type has to be an instance of DataType object!")
        self.data_type = data_type

        return self


class ForkNode(Node):
    """
    Fork node.
    """

    def __init__(self, name, children, level=1):
        """
        Initialize the ForkNode object.
        :param name: Name for the fork.
        :param children: List of Node objects.
        """
        super(ForkNode, self).__init__()
        self.level = level
        self.overwrite_children(name=name, children=children)

    def overwrite_children(self, name, children):
        """
        Force method which sets the name and the children leaves to the node.
        :param children: Array-like of Nodes.
        :param name: String specifying the name of the node.
        :return: Instance of the object itself with children and name set.
        """
        if not isinstance(children, (collections.Sequence, np.ndarray)) or isinstance(children, str):
            raise AttributeError("Incorrect format of input children nodes!")

        for child in children:
            if not isinstance(child, Node):
                raise AttributeError("Nodes have to be of Node instance!")

        self.children = children

        if not isinstance(name, str):
            raise AttributeError("The name of the node has to be a string!")

        self.name = name

        self.data_type = TreeDataType(schema=TreeSchema(base_fork_node=self))

        values, counts = np.unique(ar=self.get_children_names(), return_counts=True)
        if np.max(counts) > 1:
            raise AttributeError(
                "Children nodes with the same name are not allowed! '{}'".format(values[np.argmax(counts)]))

        return self

    def get_children(self):
        """
        Get the list of the children nodes.
        :return: List of Nodes.
        """
        if not self.is_fork():
            raise AttributeError("Cannot get children from a leaf!")

        if self.children is None:
            raise RuntimeError("Empty children leaves!")

        return self.children

    def get_children_names(self):
        """
        Get the list of children names.
        :return: List of strings representing the children names.
        """
        if not self.is_fork():
            raise AttributeError("Cannot get children from a leaf!")

        if self.children is None:
            raise RuntimeError("Empty children leaves!")

        return [x.get_name() for x in self.get_children()]

    def find_child(self, name):
        """
        Find specific child by name
        :param name: String specifying the child's name
        :return: Node
        """
        if not isinstance(name, str):
            raise AttributeError("Input parameter 'name' has to be a string!")

        child_list = [x for x in self.get_children() if x.get_name() == name]

        if not len(child_list):
            raise RuntimeError("Child '{}' was not found in '{}'".format(name, self.name))
        elif not len(child_list) - 1:
            return child_list[0]
        else:
            raise RuntimeError(
                "Impossible error achieved! More than 1 child found with the same "
                "name '{}' in Node '{}'".format(name, self.name))

    def build_value(self, value, method='numpy'):
        """
        Method which builds tree to the specific data type of the tree.
        :param value: Dictionary
        :param method: String specifying the building method (numpy or python)
        :return: Dictionary with its values casted to the correct type.
        """
        value_safe = value.copy()
        if not isinstance(value_safe, dict):
            raise RuntimeError("Incorrect input format of the value!")

        for name in value_safe.keys():
            if name not in self.get_children_names():
                raise RuntimeError("Unknown node of name '{}' not specified in the Node '{}'".format(name, self.name))

        for name in self.get_children_names():
            if method == 'numpy':
                try:
                    value_safe[name] = self.find_child(name).get_data_type().build_numpy_value(value_safe[name])
                except KeyError:
                    child_data_type = self.find_child(name).get_data_type()
                    value_safe[name] = child_data_type.build_numpy_value(child_data_type.numpy_na_value)
            elif method == 'python':
                try:
                    value_safe[name] = self.find_child(name).get_data_type().build_python_value(value_safe[name])
                except KeyError:
                    child_data_type = self.find_child(name).get_data_type()
                    value_safe[name] = child_data_type.build_python_value(child_data_type.python_na_value)
            else:
                raise AttributeError("Method '{}' is not supported!".format(method))

        return value_safe

    def __str__(self):
        return str("{}(\n" + "\t" * self.level + "{}\n" + "\t" * (self.level - 1) + " " * len(self.get_name()) + ")") \
            .format(self.get_name(), ("\n" + "\t" * self.level).join([str(x) for x in self.children]))


class ChildNode(Node):
    """
    Leaf.
    """

    def __init__(self, name, data_type):
        """
        Initialize the ChildNode object.
        :param name: Name for the child node.
        :param data_type: DataType object specifying the data type for the child.
        """
        super(ChildNode, self).__init__()

        self.overwrite_child(name=name, data_type=data_type)

    def overwrite_child(self, name, data_type):
        """
        Force method which sets the name and the data type to the node.
        :param name: String specifying the name of the node.
        :param data_type: Instance of DataType specifying the type of data for the node.
        :return: Instance of the object itself with name and data type set.
        """
        if not isinstance(name, str):
            raise AttributeError("The name of the node has to be a string!")

        self.name = name

        if not isinstance(data_type, DataType):
            raise AttributeError("Unsupported input data type: '{}'".format(data_type))

        self.data_type = data_type

        return self

    def __str__(self):
        return """{}({})""".format(self.get_name(), str(self.get_data_type()))


class TreeSchema(object):
    """
    Base class for input schema for a particular dataset.
    NB: Not a big difference between ForkNode and TreeSchema, it is important to distinguish between them though,
    since ForkNode's functionality is more tree-like, while the schema only gives more metadata about the object.
    """

    def __init__(self, base_fork_node):
        """
        Initialize the TreeSchema object (basically works as a ForkNode)
        :param base_fork_node: ForkNode containing the full tree
        """
        if not isinstance(base_fork_node, ForkNode):
            raise AttributeError("Incorrect format of input base node!")

        self.base_fork_node = base_fork_node

    def __str__(self):
        return str(self.base_fork_node)

    def create_dummy_nan_tree(self):
        """
        Create dummy tree with NaN values.
        :return: Dictionary
        """
        res = {}
        for name in self.base_fork_node.get_children_names():
            child_data_type = self.base_fork_node.find_child(name).get_data_type()
            if not isinstance(child_data_type, TreeDataType):
                res[name] = child_data_type.build_numpy_value(child_data_type.numpy_na_value)
            else:
                res[name] = child_data_type.schema.create_dummy_nan_tree()
        return res
