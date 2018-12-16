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
    DataType for trees (python dictionaries).

    :param nullable: Boolean specifying
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
        Method which converts the input value into the numpy type.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
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
        Method which converts the input value into the python type value.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
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
    Fork node.

    :param name: Name for the fork.
    :param children: List of Node objects.
    :param level: Integer specifying the level of the fork in the tree.
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
        Force method which sets the name and the children leaves to the node.
        :param children: Array-like of Nodes.
        :return: Instance of the object itself with children and name set.
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

    def find_child_in_any_branch(self, name, as_fork=False, as_copy=True):
        """
        Find specific child by name in all nodes of the fork.
        :param name: String
        :param as_fork: Boolean specifying whether the return value should be fork or not.
        :param as_copy: Boolean specifying whether the returned nodes should be a copy or direct instances.
        :return: List of Nodes
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

            if isinstance(child, ForkNode):
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
        Method which determines whether an other Fork is a sub-fork of the Fork.
        :param other: ForkNode
        :param direct: Boolean specifying whether the other has to be a direct sub-fork, i.e. not equal
        :return: Boolean
        """
        if not isinstance(other, ForkNode):
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

        return any([x.is_subtree(other) for x in self.get_children() if isinstance(x, ForkNode)])

    def _compare(self, other, method):
        if not isinstance(other, (ForkNode, ChildNode)):
            warn("Cannot compare ForkNode to {}".format(type(other)), UserWarning)
            return False

        if isinstance(other, ChildNode):
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
        elif not isinstance(other, ForkNode):
            warn("{} is a Fork while {} is a child!".format(self.get_name(), other.get_name()), UserWarning)
            return False

        if self.get_name() != other.get_name():
            warn("{} does not equal in name to {} in a fork!".format(self.get_name(), other.get_name()), UserWarning)
            return False
        else:
            return all(
                [child == other.find_child(child.get_name()) if child.get_name() in other.get_children_names() else
                 False for child in self.get_children()] + [self.level == other.level])

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
        if isinstance(other, ChildNode):
            return other * self

        # Now we are in the case where other is definitely a ForkNode. Find all the branches and children with the same
        # name as our name.
        found_branches = other.find_child_in_any_branch(self.get_name())

        # If we found at least 2 (including the root name) raise an exception
        if len(found_branches) > 1 or (len(found_branches) > 0 and self.get_name() == other.get_name()):
            raise RuntimeError("Cannot merge 2 forks by the same name! '{}'".format(self.get_name()))

        # Now do the same exercise on self, check whether the current root name does not appear in any subbranch.
        # If so raise an exception.
        if len(self.find_child_in_any_branch(self.get_name(), False, True)) > 0:
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

                # If there are none found, continue with the next child
                if not found_children:
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
        # children of the same but return them as full fork node.
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
        if isinstance(other, ChildNode):
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
                other_fork_path = other.find_child_in_any_branch(child.get_name(), as_fork=True)
                assert len(other_fork_path.get_children()) == 1

                children.append(child + other_fork_path.get_children()[0])

            # Get the names of all the children we have already unioned from self
            union_children_names = [x.get_name() for x in children]

            # For each other child
            for other_child in other.get_children():

                # If the other child's name has been already dealt with, continue to the next one
                if other_child.get_name() in union_children_names:
                    continue

                # If the other child's name appears in any subbranch of a already dealt with fork node, continue to
                # the next one
                if [x.find_child_in_any_branch(other_child.get_name()) for x in children if isinstance(x, ForkNode)]:
                    continue

                # Otherwise append this child to the children list
                children.append(deepcopy(other_child))

            # Finally return the unioned fork node
            return ForkNode(name=self.get_name(), children=children, level=self.level)

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
            if isinstance(other_child, ChildNode):
                if other_child.get_name() == self.get_name():
                    children.append(other_child + self)
                else:
                    children.append(deepcopy(other_child))

            # Otherwise we now know that other_child is a ForkNode, therefore if there is any subbranch of this fork
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
    Leaf.

    :param name: Name for the child node.
    :param data_type: DataType object specifying the data type for the child.
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

        if isinstance(other, ForkNode) and method in ('__le__', '__lt__'):
            found_children = other.find_child_in_any_branch(self.get_name())
            return any([self.__getattribute__(method.replace("t", "e"))(x) for x in found_children])
        elif isinstance(other, ForkNode) and method not in ('__le__', '__lt__'):
            return False

        if self.get_name() != other.get_name():
            return False

        return self.get_data_type().__getattribute__(method)(other.get_data_type())

    def __str__(self):
        return """{}({})""".format(self.get_name(), str(self.get_data_type()))

    def __eq__(self, other):
        if not isinstance(other, Node):
            raise AttributeError("Cannot compare ChildNode to '{}'".format(type(other)))
        elif not isinstance(other, ChildNode):
            warn("{} is a child, while {} is a fork".format(self.get_name(), other.get_name()), UserWarning)
            return False

        if self.get_name() != other.get_name():
            warn("{} does not equal in name to {} in a child!".format(self.get_name(), other.get_name()), UserWarning)
            return False
        elif self.get_data_type() != other.get_data_type():
            warn("{}'s data types are different".format(self.get_name()), UserWarning)
            return False
        else:
            return True

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
        if isinstance(other, ChildNode):
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
            if isinstance(found_children[0], ChildNode):
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
        if isinstance(other, ChildNode):

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
            if isinstance(found_child, ForkNode):
                raise ValueError(
                    "Cannot perform union on {} and {} type.".format(self.get_data_type(), found_child.get_data_type()))

            # Otherwise it is a child node so find the higher one and return the fork with path to the child
            else:
                if self <= found_child:
                    return other.find_child_in_any_branch(self.get_name(), as_fork=True)
                elif self >= found_child:
                    found_child.set_data_type(self.get_data_type())
                    return other.find_child_in_any_branch(self.get_name(), as_fork=True)

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

    :param base_fork_node: ForkNode containing the full tree
    """

    def __init__(self, base_fork_node):
        """
        Initialize the TreeSchema object (basically works as a ForkNode)
        """
        if not isinstance(base_fork_node, ForkNode):
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
        :param name: String
        :return: DataType
        """
        if not isinstance(name, str):
            raise ValueError("Parameter 'name' has to be a string!")

        arr_keys = name.split("/")
        return self._traverse(arr_keys).get_data_type()

    def set_data_type(self, name, data_type):
        """
        Method which sets the data type for the specific ndoe. The name has to be of format
        'level1-name/level2-name'...', i.e. a slash denotes forking.
        :param name: String
        :param data_type: DataType
        :return: TreeSchema
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
        Create dummy tree with NaN values.
        :param method: String speciyfing the build method
        :return: Dictionary
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
