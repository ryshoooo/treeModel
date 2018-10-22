from unittest import TestCase
import numpy as np

from src.datamodel.datatypes import DataType, StringDataType


class TestDataType(TestCase):
    """
    Test class for generic data type.
    """

    def test_is_nullable(self):
        dtp = DataType(numpy_dtype='<i4', python_dtype=int, numpy_na_value=None, python_na_value=None)
        self.assertFalse(dtp.is_nullable())
        dtp = DataType(numpy_dtype='<i4', python_dtype=int, numpy_na_value=np.nan, python_na_value=None)
        self.assertTrue(dtp.is_nullable())

    def test_get_numpy_type(self):
        dtp = DataType(numpy_dtype='<i4', python_dtype=int, numpy_na_value=None, python_na_value=None)
        self.assertEqual(dtp.get_numpy_type(), np.dtype('<i4'))
        dtp = DataType(numpy_dtype='<c16', python_dtype=float, numpy_na_value=None, python_na_value=None)
        self.assertEqual(dtp.get_numpy_type(), np.dtype('<c16'))
        dtp = DataType(numpy_dtype='<a25', python_dtype=bytes, numpy_na_value=None, python_na_value=None)
        self.assertEqual(dtp.get_numpy_type(), np.dtype('<a25'))

    def test_get_python_type(self):
        dtp = DataType(numpy_dtype='<i4', python_dtype=int, numpy_na_value=None, python_na_value=None)
        self.assertEqual(dtp.get_python_type(), int)
        dtp = DataType(numpy_dtype='<c16', python_dtype=float, numpy_na_value=None, python_na_value=None)
        self.assertEqual(dtp.get_python_type(), float)
        dtp = DataType(numpy_dtype='<a25', python_dtype=bytes, numpy_na_value=None, python_na_value=None)
        self.assertEqual(dtp.get_python_type(), bytes)

    def test_build_numpy_value(self):
        dtp = DataType(numpy_dtype='<i4', python_dtype=int, numpy_na_value=None, python_na_value=None)
        self.assertEqual(dtp.build_numpy_value(1), np.int32(1))
        dtp = DataType(numpy_dtype='<c16', python_dtype=float, numpy_na_value=None, python_na_value=None)
        self.assertEqual(dtp.build_numpy_value(1.1), np.complex128(1.1))

    def test_build_python_value(self):
        dtp = DataType(numpy_dtype='<i4', python_dtype=int, numpy_na_value=None, python_na_value=None)
        self.assertEqual(dtp.build_python_value(1), int(1))
        dtp = DataType(numpy_dtype='<c16', python_dtype=float, numpy_na_value=None, python_na_value=None)
        self.assertEqual(dtp.build_python_value(1.1), float(1.1))
        dtp = DataType(numpy_dtype='<a25', python_dtype=bytes, numpy_na_value=None, python_na_value=None)
        self.assertEqual(dtp.build_python_value(10), bytes(10))


class TestStringDataType(TestCase):
    """
    Test class for StringDataType.
    """

    def test_is_nullable(self):
        dtp = StringDataType(nullable=False)
        self.assertFalse(dtp.is_nullable())
        dtp = StringDataType(nullable=True)
        self.assertTrue(dtp.is_nullable())

    def test_get_numpy_type(self):
        dtp = StringDataType(longest_string=10)
        self.assertEqual(dtp.get_numpy_type(), np.dtype('<U10'))
        dtp = StringDataType(longest_string=1)
        self.assertEqual(dtp.get_numpy_type(), np.dtype('<U1'))
        dtp = StringDataType()
        self.assertEqual(dtp.get_numpy_type(), np.dtype('<U200'))

    def test_get_python_type(self):
        dtp = StringDataType(longest_string=10)
        self.assertEqual(dtp.get_python_type(), str)
        dtp = StringDataType(longest_string=11)
        self.assertEqual(dtp.get_python_type(), str)
        dtp = StringDataType()
        self.assertEqual(dtp.get_python_type(), str)

    def test_build_numpy_value(self):
        dtp = StringDataType(longest_string=10)
        self.assertEqual(dtp.build_numpy_value("1234567890123"), "1234567890")
        dtp = StringDataType(longest_string=1)
        self.assertEqual(dtp.build_numpy_value("123"), "1")
        dtp = StringDataType()
        self.assertEqual(dtp.build_numpy_value("tra2"), "tra2")

    def test_build_python_value(self):
        dtp = StringDataType(longest_string=10)
        self.assertEqual(dtp.build_python_value(10), "10")
        dtp = StringDataType(longest_string=1)
        self.assertEqual(dtp.build_python_value(10), "10")
        dtp = StringDataType()
        self.assertEqual(dtp.build_python_value("tra2"), "tra2")
