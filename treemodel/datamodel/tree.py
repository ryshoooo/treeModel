"""
This module contains all the tree functionality necessary for building the base tree data rows and data sets.
"""

from .datatypes import DataType
from copy import deepcopy
from functools import reduce
import collections
import numpy as np
from warnings import warn


#####################################################
#              TREE FUNCTIONALITY                   #
#####################################################

class TreeDataType(DataType):
    """
    DataType for trees.

    :param base_fork: A fork node, which specifies the expected tree structure in the input data.
    :param nullable: ``True`` or ``False`` specifying whether the input values can be missing.

    :type base_fork: ForkNode
    :type nullable: bool

    :ivar base_fork: A fork node, which specifies the expected tree structure in the input data.
    :vartype base_fork: ForkNode
    """

    def __init__(self, base_fork, nullable=True):
        """
        Initialize the data type.
        """

        if not isinstance(base_fork, ForkNode):
            raise AttributeError("Input base fork has to be an instance of ForkNode!")

        self.base_fork = base_fork

        if nullable:
            super(TreeDataType, self).__init__(dict, dict, {}, {})
        else:
            super(TreeDataType, self).__init__(dict, dict, None, None)

    def build_numpy_value(self, value):
        """
        Method which converts the input value into the numpy type. In case the underlying ``base_fork`` expects more
        entries than given in the input value, the entry will be created with a numpy missing value attribute.

        :param value: Tree-like structured input data. Must be possible to convert to a dictionary.
        :type value: dict

        :return: Converted value of the specific data type.
        :rtype: dict
        """
        value_safe = self.get_numpy_type().type(value).copy()

        if not isinstance(value_safe, dict):
            raise RuntimeError("Incorrect input format of the value!")

        for name in value_safe.keys():
            if name not in self.base_fork.get_children_names():
                raise RuntimeError(
                    "Unknown node of name '{}' not specified in the Node '{}'".format(name, self.base_fork.name))

        for name in self.base_fork.get_children_names():
            try:
                value_safe[name] = self.base_fork.find_child(name).get_data_type().build_numpy_value(value_safe[name])
            except KeyError:
                child_data_type = self.base_fork.find_child(name).get_data_type()
                value_safe[name] = child_data_type.build_numpy_value(child_data_type.numpy_na_value)

        return value_safe

    def build_python_value(self, value):
        """
        Method which converts the input value into the python type. In case the underlying ``base_fork`` expects more
        entries than given in the input value, the entry will be created with a python missing value attribute.

        :param value: Tree-like structured input data. Must be possible to convert to a dictionary.
        :type value: dict

        :return: Converted value of the specific data type.
        :rtype: dict
        """
        value_safe = self.get_python_type()(value).copy()

        if not isinstance(value_safe, dict):
            raise RuntimeError("Incorrect input format of the value!")

        for name in value_safe.keys():
            if name not in self.base_fork.get_children_names():
                raise RuntimeError(
                    "Unknown node of name '{}' not specified in the Node '{}'".format(name, self.base_fork.name))

        for name in self.base_fork.get_children_names():
            try:
                value_safe[name] = self.base_fork.find_child(name).get_data_type().build_python_value(value_safe[name])
            except KeyError:
                child_data_type = self.base_fork.find_child(name).get_data_type()
                value_safe[name] = child_data_type.build_python_value(child_data_type.python_na_value)

        return value_safe

    def _compare(self, other, method):
        if not isinstance(other, DataType):
            warn("Cannot compare TreeDataType to '{}'".format(type(other)), UserWarning)
            return False

        if isinstance(other, TreeDataType):
            return self.base_fork.__getattribute__(method)(other.base_fork)

        return any([x.get_data_type().__getattribute__(method.replace("t", "e"))(other)
                    for x in self.base_fork.get_children()])

    def __str__(self):
        return """TreeDataType({})""".format(str(self.base_fork))

    def __eq__(self, other):
        if not isinstance(other, DataType):
            raise AttributeError("Cannot compare TreeDataType to '{}'".format(type(other)))
        elif not isinstance(other, TreeDataType):
            warn("TreeDataType is not a {}".format(type(other)), UserWarning)
            return False
        else:
            return self.base_fork == other.base_fork


class Node(object):
    """
    Main node object.
    Contains all of the necessary functionality, which apply to both :class:`treemodel.datamodel.tree.ForkNode` and
    :class:`treemodel.datamodel.tree.ChildNode`. Can be also considered as a data point or a collection of data points.

    All of the subclasses of the Node class need to implement an abstract method :meth:`_compare`, which afterwards
    allows the nodes to be comparable with standard Python comparison methods (``<=``, ``>=``, etc.). Furthermore
    the addition and multiplication methods (``+``, ``*``) are also implemented for nodes, which in the tree language
    mean ``union`` and ``intersection`` of nodes in the respective order.

    :ivar children: Collection of ordered Nodes. These are the direct sub nodes of the particular node instance, only possible in :class:`treemodel.datamodel.tree.ForkNode`.
    :ivar name: The name of the node (can be considered a data point/points name as well).
    :ivar data_type: Specifies the data type of the node.

    :vartype children: list(Node)
    :vartype name: str
    :vartype data_type: DataType
    """

    def __init__(self):
        """
        Instantiation of the main node object.
        """
        # Set children nodes, name, value and the data_type of the node to None.
        self.children = None
        self.name = None
        self.data_type = None

    def is_child(self):
        """
        Simple method to determine whether the node is a leaf node (child node).

        :return: ``True`` or ``False`` based on whether the node is a child.
        :rtype: bool
        """
        return self.children is None and self.name is not None and self.data_type is not None and not isinstance(
            self.data_type, TreeDataType)

    def is_fork(self):
        """
        Simple method to determine whether the node is a fork, i.e. whether it contains more nodes.

        :return: ``True`` or ``False`` based on whether the node is a fork.
        :rtype: bool
        """
        return self.children is not None and self.name is not None and self.data_type is not None and isinstance(
            self.data_type, TreeDataType)

    def get_name(self):
        """
        Get the name of the node.

        :return: The name of the node.
        :rtype: str
        """
        if self.name is None:
            raise RuntimeError("The name of the node is missing!")

        return self.name

    def set_name(self, name):
        """
        Sets the name of the node.

        :param name: New name for the node.
        :type name: str

        :return: The same instance of the node with updated new name.
        :rtype: Node
        """
        if not isinstance(name, str):
            raise AttributeError("Parameter name has to be a string!")
        self.name = name

        return self

    def get_data_type(self):
        """
        Gets the direct instance of the DataType of the node.

        :return: The direct instance of the DataType of the node.
        :rtype: DataType
        """
        if self.data_type is None:
            raise RuntimeError("The data type is missing!")

        return self.data_type

    def set_data_type(self, data_type):
        """
        Sets the direct instance of the DataType of the node.

        :param data_type: New data type to be set.
        :type data_type: DataType

        :return: The same instance of the node with updated new data type.
        :rtype: Node
        """
        if not isinstance(data_type, DataType):
            raise AttributeError("Parameter data_type has to be an instance of DataType object!")
        self.data_type = deepcopy(data_type)

        return self

    def _compare(self, other, method):
        """
        Generic method to compare Node with other nodes.
        :param other: Node
        :param method: String
        :return: Boolean
        """
        raise NotImplementedError("Cannot compare generic Node!")

    def __le__(self, other):
        return self._compare(other, self.__le__.__name__)

    def __ge__(self, other):
        return self._compare(other, self.__ge__.__name__)

    def __lt__(self, other):
        return self._compare(other, self.__lt__.__name__)

    def __gt__(self, other):
        return self._compare(other, self.__gt__.__name__)


class ForkNode(Node):
    """
    Contains the functionality and methods specific for a forking node.

    :param name: Name for the fork.
    :param children: List of Node objects (children nodes).
    :param level: Integer specifying the level of the fork in the tree.

    :type name: str
    :type children: list(Node)
    :type level: int

    :ivar level: Integer specifying the level of the fork in the tree.
    :vartype level: int
    """

    def __init__(self, name, children, level=1):
        """
        Initialize the ForkNode object.
        """
        super(ForkNode, self).__init__()
        self.level = level

        self.set_name(name=name)
        self.set_children(children=children)
        self.data_type = TreeDataType(base_fork=self)

    def set_children(self, children):
        """
        Force setter method which sets the children leaves to the current forking node instance.

        :param children: Array-like of Nodes, which are to be set as children.
        :type children: list(Node)

        :return: Instance of the object itself with new set of children set.
        :rtype: ForkNode
        """
        if not isinstance(children, (collections.Sequence, np.ndarray)) or isinstance(children, str):
            raise AttributeError("Incorrect format of input children nodes!")

        for child in children:
            if not isinstance(child, Node):
                raise AttributeError("Nodes have to be of Node instance!")

        self.children = deepcopy(children)

        values, counts = np.unique(ar=[x.get_name() for x in children], return_counts=True)

        if len(counts) != 0 and np.max(counts) > 1:
            raise AttributeError(
                "Children nodes with the same name are not allowed! '{}'".format(values[np.argmax(counts)]))

        return self

    def get_children(self):
        """
        Gets the list of the children nodes of the current fork instance.

        :return: List of children nodes.
        :rtype: list(Node)
        """
        if not self.is_fork():
            raise AttributeError("Cannot get children from a leaf!")

        if self.children is None:
            raise RuntimeError("Empty children leaves!")

        return self.children

    def get_children_names(self):
        """
        Gets the list of children names in the children order of the current fork instance.

        :return: List of the children names.
        :rtype: list(str)
        """
        if not self.is_fork():
            raise AttributeError("Cannot get children from a leaf!")

        if self.children is None:
            raise RuntimeError("Empty children leaves!")

        return [x.get_name() for x in self.get_children()]

    def find_child(self, name):
        """
        Finds specific child by name in the current fork level only.

        :param name: Specifies the child's name to be found.
        :type name: String

        :raises: :class:`RuntimeError` in case the wanted child is not found.

        :return: Direct instance of the child with the wanted name.
        :rtype: Node
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

    def find_child_in_any_branch(self, name, as_fork=False, as_copy=True):
        """
        Finds all of the nodes bearing the wanted name in all of the nodes of the fork.

        :param name: Name of the children to be found.
        :param as_fork: ``True`` or ``False`` whether the return value should be an instance of the ``ForkNode``.
        :param as_copy: ``True`` or ``False`` specifying whether the returned nodes should be a copy or direct instances.

        :type name: str
        :type as_fork: bool
        :type as_copy: bool

        :return: Either list of nodes carrying the wanted name, or a ForkNode with the full paths from the current fork node to the children carrying the wanted name, or None in case none children are found.
        :rtype: list(Node) or ForkNode or None
        """
        if not isinstance(name, str):
            raise ValueError("Input parameter 'name' has to be a string")

        res = []
        for child in self.get_children():
            if child.get_name() == name:
                if as_copy:
                    res.append(deepcopy(child))
                else:
                    res.append(child)

            if child.is_fork():
                found_children = child.find_child_in_any_branch(name, as_fork, as_copy)

                if as_fork and found_children is not None:
                    found_children = [found_children]
                if as_fork and found_children is None:
                    found_children = []

                res += found_children

        if as_fork and not res:
            return None
        if as_fork:
            return ForkNode(self.get_name(), res, self.level)
        else:
            return res

    def is_subtree(self, other, direct=False):
        """
        Method which determines whether an other Fork is a sub-fork of the current fork node instance.
        A fork node (other) is a ``subfork`` of an another fork node (self) if and only if other fork node appears in
        the self fork with the exact correct levelling.

        :param other: A fork which is to be determined whether is a subtree of the current fork instance.
        :param direct: ``True`` or ``False`` specifying whether the other has to be a direct sub-fork, i.e. not equal to the current instance of the fork.

        :type other: ForkNode
        :type direct: bool

        :raises: :class:`ValueError` in case parameter ``other`` is not a fork.

        :return: ``True`` or ``False`` specifying whether the other fork is a subfork of the current instance.
        :rtype: bool
        """
        if not other.is_fork():
            raise ValueError("Parameter other has to be an instance of ForkNode! '{}'".format(type(other)))

        if self.get_name() == other.get_name():
            other_children = other.get_children()
            try:
                res = all([other_child <= self.find_child(other_child.get_name()) for other_child in other_children])
                if res and not direct:
                    return True
                elif res and direct:
                    return not self == other
                else:
                    return res
            except RuntimeError:
                return False

        return any([x.is_subtree(other) for x in self.get_children() if x.is_fork()])

    def _compare(self, other, method):
        if not isinstance(other, (ForkNode, ChildNode)):
            warn("Cannot compare ForkNode to {}".format(type(other)), UserWarning)
            return False

        if other.is_child():
            if 'g' in method:
                method = method.replace("g", "l")
            else:
                method = method.replace("l", "g")
            return other.__getattribute__(method)(self)

        if 'g' in method:
            return self.is_subtree(other, 'e' not in method)
        else:
            return other.is_subtree(self, 'e' not in method)

    def __str__(self):
        """
        Format the ForkNode into a string, add tab for each child based on the level of the fork.
        :return: String
        """
        children_str = ("\n" + "\t" * self.level).join([str(x) for x in self.children])
        return str("{}(\n" + "\t" * self.level + "{}\n" + "\t" * (self.level - 1) + " " * len(self.get_name()) + ")") \
            .format(self.get_name(), children_str)

    def __eq__(self, other):
        if not isinstance(other, Node):
            raise AttributeError("Cannot compare ForkNode with '{}'".format(type(other)))
        elif not other.is_fork():
            warn("{} is a Fork while {} is a child!".format(self.get_name(), other.get_name()), UserWarning)
            return False

        if self.get_name() != other.get_name():
            warn("{} does not equal in name to {} in a fork!".format(self.get_name(), other.get_name()), UserWarning)
            return False
        else:
            try:
                return all([child == other.find_child(child.get_name()) for child in self.get_children()] +
                           [self.level == other.level]) \
                       and all([child == self.find_child(child.get_name()) for child in other.get_children()])
            except RuntimeError:
                return False

    def __mul__(self, other, _iter=1):
        """
        Performs a uniformized intersection on 2 Nodes.
        :param other: Node, either a ChildNode or a ForkNode
        :param _iter: Int, helper variable notifying a recursive call on the same method in the same class.
        :return: Node or None
        """
        # First check the type of the input variable
        if not isinstance(other, (ChildNode, ForkNode)):
            raise ValueError("Intersection is not defined for type '{}'".format(type(other)))

        # In case of ChildNode, call the child node implementation
        if other.is_child():
            return other * self

        # Now we are in the case where other is definitely a ForkNode. Find all the branches and children with the same
        # name as our name.
        found_branches = other.find_child_in_any_branch(self.get_name())

        # If we found at least 2 (including the root name) raise an exception
        if len(found_branches) > 1 or (len(found_branches) > 0 and self.get_name() == other.get_name()):
            raise RuntimeError("Cannot merge 2 forks by the same name! '{}'".format(self.get_name()))

        # Now do the same exercise on self, check whether the current root name does not appear in any subbranch.
        # If so raise an exception.
        if self.find_child_in_any_branch(self.get_name()):
            raise RuntimeError("Cannot merge 2 forks by the same name! '{}'".format(self.get_name()))

        # Now in case the root names are the same, begin the intersection process
        if self.get_name() == other.get_name():

            # Check that each other child name does not appear more than once in our fork. If so raise exception.
            for other_child_name in other.get_children_names():
                if len(self.find_child_in_any_branch(other_child_name)) > 1:
                    raise RuntimeError(
                        "Cannot merge 2 forks with children of the same name! '{}'".format(other_child_name))

            # Initialize the merged children list and for each child of our own do the following
            children = []
            for child in self.get_children():
                # Find all the children of the same name in any subbranch of the other fork
                found_children = other.find_child_in_any_branch(child.get_name(), as_fork=False)

                # If there are more than 1 found, raise an Exception (impossible merging)
                if len(found_children) > 1:
                    raise RuntimeError(
                        "Cannot merge 2 forks with children of the same name! '{}'".format(child.get_name()))

                # If there are none found and we have a leaf, continue with the next child
                if not found_children and child.is_child():
                    continue

                # Now we are in the case where only 1 child was found. However it does not have to be a direct child
                # of the current fork, since we were searching through any subbranch. Therefore we need to identify
                # which child contains our wanted result. Therefore intersect the current child with every single
                # other child and filter the ones which are None.
                merged_children = [child * other_child for other_child in other.get_children()]
                merged_children = [x for x in merged_children if x is not None]

                # Only 1 single element should be in this list due to the checks before the merging. Therefore if there
                # are more than 1 element in the merged children list, an impossible error was achieved, which can only
                # mean a disruptive change to the code. Thus raise exception.
                if len(merged_children) > 1:
                    raise RuntimeError("Impossible error achieved, congratulations!")

                # Append the child to the merged children's list.
                children += merged_children

            # After the for iterations, if there are no merged children, then the intersection resulted into nothing
            # and thus return None. Otherwise return the intersected ForkNode.
            if not children:
                return None
            else:
                return ForkNode(name=self.get_name(), children=children, level=self.level)

        # Now we are in the case where intersection might be possible but the root names do not match.
        # If there are no found branches in other fork, try to perform the intersection other * self instead.
        # This will be only performed once due to _iter variable being raised to 2. If the same case happens
        # for reversed intersection, return None since there is nothing to be merged together.
        if not found_branches:
            if _iter == 1:
                return other.__mul__(self, 2)
            else:
                return None

        # Now we are in the case when there is something to be merged together, i.e. found branches list is not empty.
        # That particularly means that the other fork node is `larger` than ourselves. Thus find once again all the
        # children of the same name but return them as full fork node.
        larger_fork = other.find_child_in_any_branch(self.get_name(), as_fork=True)
        # Then merge each other child with self and filter all the empty ones.
        larger_forks_merged_children = [x * self for x in larger_fork.get_children()]
        larger_forks_merged_children = [x for x in larger_forks_merged_children if x is not None]

        # Now if after filtering there is nothing left, then the intersection with smaller subbranches of other was
        # not successful and we shall return nothing. On the other hand if at least one intersection with the
        # children was successful, we have something to return. Set the children of the larger fork to the intersected
        # ones and return.
        if not larger_forks_merged_children:
            return None
        else:
            return larger_fork.set_children(larger_forks_merged_children)

    def __add__(self, other, _iter=1):
        """
        Performs a uniformized union on 2 Nodes.
        :param other: Node, either a ChildNode or a ForkNode
        :param _iter: Int, helper variable notifying a recursive call on the same method in the same class.
        :return: ForkNode
        """
        # First check the type of the input variable
        if not isinstance(other, (ChildNode, ForkNode)):
            raise ValueError("Union is not defined for type '{}'".format(type(other)))

        # In case of ChildNode, call the child node implementation
        if other.is_child():
            return other.__add__(self, self.level)

        # Now we are in the case where other is definitely a ForkNode. Find all the branches and children with the same
        # name as our name.
        found_branches = other.find_child_in_any_branch(self.get_name())

        # If we found at least 2 (including the root name) raise an exception
        if len(found_branches) > 1 or (len(found_branches) > 0 and self.get_name() == other.get_name()):
            raise RuntimeError("Cannot union 2 forks by the same name! '{}'".format(self.get_name()))

        # Now do the same exercise on self, check whether the current root name does not appear in any subbranch.
        # If so raise an exception.
        if len(self.find_child_in_any_branch(self.get_name())) > 0:
            raise RuntimeError("Cannot union 2 forks by the same name! '{}'".format(self.get_name()))

        # Now in case the root names are the same, begin the union process
        if self.get_name() == other.get_name():

            # Check that each other child name does not appear more than once in our fork. If so raise exception.
            for other_child_name in other.get_children_names():
                if len(self.find_child_in_any_branch(other_child_name)) > 1:
                    raise RuntimeError(
                        "Cannot union 2 forks with children of the same name! '{}'".format(other_child_name))

            # Initialize the merged children list and for each child of our own do the following
            children = []

            for child in self.get_children():
                # Find all the children of the same name in any subbranch of the other fork
                found_children = other.find_child_in_any_branch(child.get_name())

                # If there are more than 1 found, raise an Exception (impossible for union)
                if len(found_children) > 1:
                    raise RuntimeError(
                        "Cannot union 2 forks with children of the same name! '{}'".format(child.get_name()))

                # If there are none found, append this child to the children and continue with the next child
                if not found_children:
                    children.append(deepcopy(child))
                    continue

                # Now we are in the case of having only 1 single child found
                found_child = found_children[0]

                # If the current child appears directly in the other children, perform union on them and append
                if child.get_name() in other.get_children_names():
                    children.append(child + found_child)
                    continue

                # Otherwise the found child in other is not a direct child of the root, therefore perform union
                # on the current child and the fork leading to the child in the other subbranch. Append the result
                # to the children list.
                other_forks = [other_child for other_child in other.get_children() if other_child.is_fork()]
                potentials = [other_fork for other_fork in other_forks if
                              other_fork.find_child_in_any_branch(child.get_name())]
                assert len(potentials) == 1
                other_fork_path = potentials[0]

                children.append(child + other_fork_path)

            # Get the names of all the children we have already unioned from self
            union_children_names = [x.get_name() for x in children]

            # For each other child
            for other_child in other.get_children():

                # If the other child's name has been already dealt with, continue to the next one
                if other_child.get_name() in union_children_names:
                    continue

                # If the other child's name appears in any subbranch of a already dealt with fork node, continue to
                # the next one
                if [x.find_child_in_any_branch(other_child.get_name()) for x in children if x.is_fork()]:
                    continue

                # Otherwise append this child to the children list
                children.append(deepcopy(other_child))

            # Finally return the unioned fork node
            return ForkNode(name=self.get_name(), children=children, level=max(self.level, other.level))

        # In case there are no found subbranches from the other fork
        if not found_branches:
            # In case this is the first iteration, perform the union in reversed order
            if _iter == 1:
                return other.__add__(self, 2)
            else:  # Otherwise this is the second union iteration, thus merge these 2 forks together with proper levels
                other_res = deepcopy(other)
                other_res.level += 1
                self_res = deepcopy(self)
                self_res.level = other_res.level

                return ForkNode(name="base_{}_{}".format(other_res.get_name(), self_res.get_name()),
                                children=[other_res, self_res], level=other.level)

        # Now we are in a case where the names of the forks do not match but there is a subbranch in other fork node
        # which carries the same name as self node. Therefore the union is going to be based on the other fork node.
        # First initialize the children empty list.
        children = []

        # Now for each other child in the other fork node do the following
        for other_child in other.get_children():

            # In case the other child is a ChildNode and the child node carries the same name as our fork,
            # this will result in exception, which is implemented in the ChildNode union method. Otherwise if the
            # names differ, append the other child to the children list.
            if other_child.is_child():
                if other_child.get_name() == self.get_name():
                    children.append(other_child + self)
                else:
                    children.append(deepcopy(other_child))
            # Now we are in the case where the other child is a fork. If it shares the same name as we do, perform
            # the union and append as a new child to the children's list.
            elif other_child.get_name() == self.get_name():
                children.append(self + other_child)
            # Otherwise we now know that other_child is a ForkNode and that there has to be a subbranch carrying the
            # same name as we do, therefore if there is any subbranch of this fork
            # node, which carries the same name as self node does, union them together and append the result to the
            # children list.
            elif other_child.find_child_in_any_branch(self.get_name()):
                children.append(self + other_child.find_child_in_any_branch(self.get_name(), as_fork=True))
            # Otherwise just append the child to the children list.
            else:
                children.append(deepcopy(other_child))

        # Finally return the union of the 2 fork nodes
        return ForkNode(name=other.get_name(), children=children, level=other.level)


class ChildNode(Node):
    """
    Implementation of the leaf node, carries only information about the data type of the leaf and its name.

    :param name: Name for the child node.
    :param data_type: Specifying the data type for the leaf.

    :type name: str
    :type data_type: DataType
    """

    def __init__(self, name, data_type):
        """
        Initialize the ChildNode object.
        """
        super(ChildNode, self).__init__()

        self.set_name(name=name)
        self.set_data_type(data_type=data_type)

    def _compare(self, other, method):
        if not isinstance(other, (ForkNode, ChildNode)):
            warn("Cannot compare ChildNode to {}".format(type(other)), UserWarning)
            return False

        if other.is_fork() and method in ('__le__', '__lt__'):
            found_children = other.find_child_in_any_branch(self.get_name())
            return any([self.__getattribute__(method.replace("t", "e"))(x) for x in found_children])
        elif other.is_fork() and method not in ('__le__', '__lt__'):
            return False

        if self.get_name() != other.get_name():
            return False

        return self.get_data_type().__getattribute__(method)(other.get_data_type())

    def __str__(self):
        return """{}({})""".format(self.get_name(), str(self.get_data_type()))

    def __eq__(self, other):
        if not isinstance(other, Node):
            raise AttributeError("Cannot compare ChildNode to '{}'".format(type(other)))
        elif not other.is_child():
            warn("{} is a child, while {} is a fork".format(self.get_name(), other.get_name()), UserWarning)
            return False

        if self.get_name() != other.get_name():
            warn("{} does not equal in name to {} in a child!".format(self.get_name(), other.get_name()), UserWarning)
            return False
        else:
            return self.get_data_type() == other.get_data_type()

    def __mul__(self, other):
        """
        Perform intersection on child nodes.
        :param other: Node (ForkNode or ChildNode)
        :return: Node
        """
        # First verify that we indeed have the correct input other type
        if not isinstance(other, (ChildNode, ForkNode)):
            raise ValueError("Cannot perform intersection on object of type '{}'!".format(type(other)))

        # In case of other being child node
        if other.is_child():
            # Find the higher one and return it
            if self <= other:
                return deepcopy(other)
            elif self >= other:
                return deepcopy(self)
            else:  # In case of incomparable data types, return nothing
                return None

        # Now we are in the case of having other a ForkNode. If the name of our child and the other fork are the same,
        # we cannot intersect fork and a child node so return nothing.
        if self.get_name() == other.get_name():
            return None

        # Otherwise find all the subbranches carrying the same name as our child
        found_children = other.find_child_in_any_branch(self.get_name(), as_fork=False, as_copy=False)

        # If there are no found, return nothing
        if not found_children:
            return None

        # If there is exactly 1 match, then do the following
        if len(found_children) == 1:

            # If what we found is a child node, then find a higher one and return it with the full fork path
            if found_children[0].is_child():
                found_child = found_children[0]
                if self <= found_child:
                    return other.find_child_in_any_branch(self.get_name(), as_fork=True)
                elif self >= found_child:
                    found_child.set_data_type(self.get_data_type())
                    return other.find_child_in_any_branch(self.get_name(), as_fork=True)
                else:  # In case of incomparable data types, return nothing
                    return None
            else:  # Otherwise we found a fork, which means to return nothing
                return None

        # Finally in case of having more than 1 subbranch found in other, raise exception
        if len(found_children) > 1:
            raise RuntimeError(
                "Cannot perform intersection where there are multiple children of the same name: '{}'".format(
                    self.get_name()))

    def __add__(self, other, level=1):
        """
        Perform union on child nodes.
        :param other: Node (ChildNode or a ForkNode)
        :param level: Int specifying the recursive level invocation
        :return: Node
        """
        # First verify that we indeed have the correct input other type
        if not isinstance(other, (ChildNode, ForkNode)):
            raise ValueError("Cannot perform union on object of type '{}'!".format(type(other)))

        # In case of other being child node
        if other.is_child():

            # Find the higher one and return it
            if self <= other:
                return deepcopy(other)
            elif self >= other:
                return deepcopy(self)

            # In case they are incomparable but share the same name, raise exception
            elif self.get_name() == other.get_name():
                raise ValueError(
                    "Cannot perform union on {} and {} type".format(self.get_data_type(), other.get_data_type()))

            # Otherwise union them into single fork and return the fork
            else:
                return ForkNode(name="base_{}_{}".format(self.get_name(), other.get_name()),
                                children=[deepcopy(self), deepcopy(other)], level=level)

        # Now we are in the case where other is a ForkNode. If it shares the same name as our current child node,
        # raise exception.
        if self.get_name() == other.get_name():
            raise ValueError(
                "Cannot perform union on {} and {} type.".format(self.get_data_type(), other.get_data_type()))

        # Find all the subbranches with the same name as our child
        found_children = other.find_child_in_any_branch(self.get_name(), as_fork=False, as_copy=False)

        # If there are none found, gather all the other children and return a fork node with them, including
        # our own self child node.
        if not found_children:
            o_name, o_children = other.get_name(), deepcopy(other.get_children())
            return ForkNode(name=o_name, children=o_children + [deepcopy(self)], level=other.level)

        # In case there is exactly 1 match found, then do the following
        if len(found_children) == 1:

            # Get the found child first
            found_child = found_children[0]

            # In case it is a fork node, raise exception
            if found_child.is_fork():
                raise ValueError(
                    "Cannot perform union on {} and {} type.".format(self.get_data_type(), found_child.get_data_type()))

            # Otherwise it is a child node so find the higher one and return the fork with path to the child
            else:
                if self <= found_child:
                    return deepcopy(other)
                elif self >= found_child:
                    found_child.set_data_type(self.get_data_type())
                    return deepcopy(other)

                # In case they are incomparable, raise exception (union is not possible)
                else:
                    raise ValueError(
                        "Cannot perform union on {} and {} type.".format(self.get_data_type(),
                                                                         found_child.get_data_type()))

        # Otherwise in case we have found more than 1 subbranch in other, raise exception
        elif len(found_children) > 1:
            raise RuntimeError(
                "Cannot perform union where there are multiple children of the same name: '{}'".format(
                    self.get_name()))


class TreeSchema(object):
    """
    Base class for input schema for a particular dataset.

    NB: Not a big difference between ForkNode and TreeSchema, it is important to distinguish between them though,
    since ForkNode's functionality is more tree-like, while the schema only gives more metadata about the object.

    :param base_fork_node: A fork node which fully specifies the expected tree schema.
    :type base_fork_node: ForkNode
    """

    def __init__(self, base_fork_node):
        """
        Initialize the TreeSchema object (basically works as a ForkNode)
        """
        if not base_fork_node.is_fork():
            raise AttributeError("Incorrect format of input base node!")

        self.base_fork_node = base_fork_node

    def _traverse(self, arr_keys):
        """
        Helper method which traverses through the tree.
        :param arr_keys: List of strings representing the keys in order to traverse through.
        :return: Node
        """
        return reduce(lambda x, y: x.find_child(y), arr_keys, self.base_fork_node)

    def find_data_type(self, name):
        """
        Method which finds the data type for the specific node. The name has to be of format
        'level1-name/level2-name/...', i.e. a slash denotes forking.

        :Example:

        >>> from treemodel.datamodel.datatypes import StringDataType
        >>> from treemodel.datamodel.tree import TreeSchema, ForkNode, ChildNode
        >>> level2_fork = ForkNode(name='example', children=[ChildNode('child', StringDataType())], level=2)
        >>> level1_fork = ForkNode(name='base', children=[level2_fork], level=1)
        >>> ts = TreeSchema(base_fork_node=level1_fork)
        >>>
        >>> print('Tree Schema: {}'.format(ts))
        >>> print("Data Type of 'child' leaf: {}".format(ts.find_data_type('example/child')))
        >>> print("Data Type of 'example' node: {}".format(ts.find_data_type('example')))

        :param name: Nested name of a node in the schema.
        :type name: str

        :return: The data type of the wanted node.
        :rtype: DataType
        """
        if not isinstance(name, str):
            raise ValueError("Parameter 'name' has to be a string!")

        arr_keys = name.split("/")
        return self._traverse(arr_keys).get_data_type()

    def set_data_type(self, name, data_type):
        """
        Method which sets the data type for the specific node. The name has to be of format
        'level1-name/level2-name'...', i.e. a slash denotes forking.

        :param name: Nested name of the wanted node.
        :param data_type: New data type to be set for the wanted nested node.

        :type name: str
        :type data_type: DataType

        :return: An instance of the current TreeSchema with the wanted node set to a new data type.
        :rtype: TreeSchema
        """
        if not isinstance(name, str):
            raise ValueError("Parameter 'name' has to be a string!")
        if not isinstance(data_type, DataType):
            raise ValueError("Parameter 'data_type' has to be a DataType!")

        arr_keys = name.split("/")
        self._traverse(arr_keys).set_data_type(data_type)

        return self

    def create_dummy_nan_tree(self, method='numpy'):
        """
        Creates a dummy tree with missing values only based on the current instance's schema.

        :param method: Specifies the method to build the tree of missing values, either ``'python'`` or ``'numpy'``.
        :type method: str

        :return: Tree-like structure of the instance's schema filled with missing values.
        :rtype: dict
        """
        if method == 'numpy':
            return self.base_fork_node.get_data_type().build_numpy_value(value={})
        elif method == 'python':
            return self.base_fork_node.get_data_type().build_python_value(value={})
        else:
            raise ValueError("Unknown method: '{}'".format(method))

    def __str__(self):
        """
        String method on schema.
        :return: String
        """
        return str(self.base_fork_node)

    def __eq__(self, other):
        """
        Equality method on schema.
        :param other: TreeSchema
        :return: Boolean
        """
        if not isinstance(other, TreeSchema):
            raise AttributeError("Cannot compare TreeSchema with '{}'".format(type(other)))

        return self.base_fork_node == other.base_fork_node

    def __mul__(self, other):
        """
        Multiplication method on schema, i.e. intersection of 2 schemas.
        :param other: TreeSchema
        :return: TreeSchema
        """
        if not isinstance(other, TreeSchema):
            raise ValueError("Intersection of schemas can be performed only on TreeSchema objects!")

        merged_base_forks = self.base_fork_node * other.base_fork_node

        if merged_base_forks is None:
            return TreeSchema(base_fork_node=ForkNode('empty', []))
        else:
            return TreeSchema(base_fork_node=merged_base_forks)

    def __add__(self, other):
        """
        Addition method on schema, i.e. union of 2 schemas.
        :param other: TreeSchema
        :return: TreeSchema
        """
        if not isinstance(other, TreeSchema):
            raise ValueError("Intersection of schemas can be performed only on TreeSchema objects!")

        return TreeSchema(base_fork_node=self.base_fork_node + other.base_fork_node)
