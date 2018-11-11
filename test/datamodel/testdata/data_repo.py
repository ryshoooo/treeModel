"""
This module serves as a data repository for tests.
"""

import numpy as np

from src.datamodel.datatypes import StringDataType, FloatDataType, ArrayDataType, ListDataType
from src.datamodel.tree import ChildNode, ForkNode, TreeDataType, TreeSchema


class DataGenerator(object):
    """
    Generator and repository of data used for testing.
    """

    @staticmethod
    def base_dict_json_same_schema():
        d = {
            "level1-string": str(np.random.choice(["A", "B", "C", "D", "R"], replace=False)),
            "level1-float": float(np.random.random()),
            "level1-date": str(np.random.choice(["{}-04-01".format(year) for year in range(1993, 2019)])),
            "level1-array_float": [float(x) for x in np.random.random(10)],
            "level1-array_string": [str(x) for x in
                                    np.random.choice(a=["A", "B", "C", "D", "R"], size=10, replace=True)],
            "level1-list_float_string": [float(x) for x in np.random.random(5)] +
                                        [str(x) for x in
                                         np.random.choice(a=["A", "B", "C", "D", "R"], size=5, replace=True)],
            "level1-fork": {
                "level2-string": str(np.random.choice(["A", "B", "C", "D", "R"], replace=False)),
                "level2-float": float(np.random.random()),
                "level2-date": str(np.random.choice(["{}-04-01".format(year) for year in range(1993, 2019)])),
                "level2-array_float": [float(x) for x in np.random.random(10)],
                "level2-array_string": [str(x) for x in
                                        np.random.choice(a=["A", "B", "C", "D", "R"], size=10, replace=True)],
                "level2-list_float_string": [float(x) for x in np.random.random(5)] +
                                            [str(x) for x in
                                             np.random.choice(a=["A", "B", "C", "D", "R"], size=5,
                                                              replace=True)],
            },
            "level1-fork2": {
                "level2-float": float(np.random.random()),
                "level2-fork": {
                    "level3-float": float(np.random.random()),
                    "level3-array_tree": [
                        {
                            "level3-array-float": float(np.random.random()),
                            "level3-array-string": str(
                                np.random.choice(["A", "B", "C", "D", "R"], replace=False))
                        } for x in range(10)
                    ],
                    "level3-list_tree": [
                                            {
                                                "level3-list-float": float(np.random.random()),
                                                "level3-list-string": str(
                                                    np.random.choice(["A", "B", "C", "D", "R"], replace=False))
                                            } for x in range(5)
                                        ] + [
                                            {
                                                "level3-list-date": str(np.random.choice(
                                                    ["{}-04-01".format(year) for year in range(1993, 2019)])),
                                                "level3-list-string": str(
                                                    np.random.choice(["A", "B", "C", "D", "R"], replace=False))
                                            } for x in range(5)
                                        ]
                }
            }
        }
        return d

    @staticmethod
    def base_dict_json_same_schema_types():
        d = {
            "level1-string": StringDataType(),
            "level1-float": FloatDataType(),
            "level1-date": StringDataType(),
            "level1-array_float": ArrayDataType(FloatDataType()),
            "level1-array_string": ArrayDataType(StringDataType()),
            "level1-list_float_string": ListDataType([FloatDataType()] * 5 + [StringDataType()] * 5),
            "level1-fork": {
                "level2-string": StringDataType(),
                "level2-float": FloatDataType(),
                "level2-date": StringDataType(),
                "level2-array_float": ArrayDataType(FloatDataType()),
                "level2-array_string": ArrayDataType(StringDataType()),
                "level2-list_float_string": ListDataType([FloatDataType()] * 5 + [StringDataType()] * 5),
            },
            "level1-fork2": {
                "level2-float": FloatDataType(),
                "level2-fork": {
                    "level3-float": FloatDataType(),
                    "level3-array_tree": ArrayDataType(
                        TreeDataType(
                            base_fork=ForkNode(
                                name="level3-array_tree",
                                children=[
                                    ChildNode(name="level3-array-float", data_type=FloatDataType()),
                                    ChildNode(name="level3-array-string", data_type=StringDataType())
                                ]
                            )
                        )
                    ),
                    "level3-list_tree": ListDataType(
                        [
                            TreeDataType(
                                base_fork=ForkNode(
                                    name="level3-list_tree_{}".format(x),
                                    children=[
                                        ChildNode(name="level3-list-float", data_type=FloatDataType()),
                                        ChildNode(name="level3-list-string", data_type=StringDataType())
                                    ]
                                )
                            )
                            for x in range(0, 5)] + [
                            TreeDataType(
                                base_fork=ForkNode(
                                    name="level3-list_tree_{}".format(x),
                                    children=[
                                        ChildNode(name="level3-list-date", data_type=StringDataType()),
                                        ChildNode(name="level3-list-string", data_type=StringDataType())
                                    ]
                                )
                            )
                            for x in range(5, 10)]
                    )
                }
            }
        }
        return d

    @staticmethod
    def simple_dict_for_print_v1():
        input_row = {"level1-float": 12.2,
                     "level1-list": ["s", 2],
                     'level1-fork': {'level2-string': 'wrq2',
                                     'level2-array': [{"array_tree_0": 0, "array_tree_1": "sd"}, {"b": 1}]},
                     "level1": "OK"}

        expected_output = """base(
	level1(StringDataType)
	level1-float(FloatDataType)
	level1-fork(
		level2-array(ListDataType(
			TreeDataType(level2-array_0(
				array_tree_0(FloatDataType)
				array_tree_1(StringDataType)
			              ))
			TreeDataType(level2-array_1(
				b(FloatDataType)
			              ))
		            ))
		level2-string(StringDataType)
	           )
	level1-list(ListDataType(
		StringDataType
		FloatDataType
	            ))
    )"""

        return input_row, expected_output

    @staticmethod
    def simple_dict_for_print_v2():
        input_row = {"level1-float": 12.2,
                     "level1-list": ["s", 2],
                     'level1-fork': {'level2-string': 'wrq2',
                                     'level2-array': [{"b": 2}, {"b": 1}]},
                     "level1": "OK"}

        expected_output = """base(
	level1(StringDataType)
	level1-float(FloatDataType)
	level1-fork(
		level2-array(ArrayDataType(TreeDataType(level2-array(
				b(FloatDataType)
			            ))))
		level2-string(StringDataType)
	           )
	level1-list(ListDataType(
		StringDataType
		FloatDataType
	            ))
    )"""

        return input_row, expected_output

    @staticmethod
    def sample_dict_for_test_schema_v1():
        input_dict = {
            'a': 23,
            'b': {
                'c': "sa",
                'd': [{"s": 1}, 12.3],
                'e': ["a", "b", "c"]
            }
        }

        expected_output = TreeSchema(base_fork_node=ForkNode(name="base", children=[
            ChildNode(name="a", data_type=FloatDataType()),
            ForkNode(name="b", children=[
                ChildNode(name="c", data_type=StringDataType()),
                ChildNode(name="d", data_type=ListDataType(element_data_types=[
                    TreeDataType(
                        base_fork=ForkNode(name="d_0", children=[ChildNode(name="s", data_type=FloatDataType())],
                                           level=4)),
                    FloatDataType()
                ], level=3)),
                ChildNode(name="e", data_type=ArrayDataType(element_data_type=StringDataType()))
            ], level=2)
        ], level=1))

        return input_dict, expected_output
