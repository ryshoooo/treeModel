from unittest import TestCase
import numpy as np
from datetime import datetime

from src.datamodel.datatypes import DataType, StringDataType, FloatDataType, DateDataType


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


class TestFloatDataType(TestCase):
    """
    Test class for FloatDataType.
    """

    def test_is_nullable(self):
        dtp = FloatDataType(nullable=False)
        self.assertFalse(dtp.is_nullable())
        dtp = FloatDataType(nullable=True)
        self.assertTrue(dtp.is_nullable())

    def test_get_numpy_type(self):
        dtp = FloatDataType(bits=2)
        self.assertEqual(dtp.get_numpy_type(), np.dtype('<f2'))
        dtp = FloatDataType(bits=4)
        self.assertEqual(dtp.get_numpy_type(), np.dtype('<f4'))
        dtp = FloatDataType()
        self.assertEqual(dtp.get_numpy_type(), np.dtype('<f8'))

    def test_get_python_type(self):
        dtp = FloatDataType(bits=2)
        self.assertEqual(dtp.get_python_type(), float)
        dtp = FloatDataType(bits=4)
        self.assertEqual(dtp.get_python_type(), float)
        dtp = FloatDataType()
        self.assertEqual(dtp.get_python_type(), float)

    def test_build_numpy_value(self):
        dtp = FloatDataType(bits=2)
        self.assertEqual(dtp.build_numpy_value("12"), 12)
        dtp = FloatDataType(bits=4)
        self.assertEqual(dtp.build_numpy_value("12.3"), np.float32(12.3))
        dtp = FloatDataType()
        self.assertEqual(dtp.build_numpy_value(9.99), 9.99)

    def test_build_python_value(self):
        dtp = FloatDataType(bits=2)
        self.assertEqual(dtp.build_python_value("12"), 12)
        dtp = FloatDataType(bits=4)
        self.assertEqual(dtp.build_python_value("12.3"), float(12.3))
        dtp = FloatDataType()
        self.assertEqual(dtp.build_python_value(9.99), 9.99)


class TestDateDataType(TestCase):
    """
    Test class for StringDataType.
    """

    def test_is_nullable(self):
        dtp = DateDataType(nullable=False)
        self.assertFalse(dtp.is_nullable())
        dtp = DateDataType(nullable=True)
        self.assertTrue(dtp.is_nullable())

    def test_get_numpy_type(self):
        dtp = DateDataType(resolution='Y')
        self.assertEqual(dtp.get_numpy_type(), np.dtype('<M8[Y]'))
        dtp = DateDataType(resolution='M')
        self.assertEqual(dtp.get_numpy_type(), np.dtype('<M8[M]'))
        dtp = DateDataType()
        self.assertEqual(dtp.get_numpy_type(), np.dtype('<M8[s]'))

    def test_get_python_type(self):
        dtp = DateDataType(resolution='Y', format_string="%Y")
        self.assertEqual(type(dtp.get_python_type()("2018")), datetime)
        dtp = DateDataType(resolution='M', format_string="%Y")
        self.assertEqual(type(dtp.get_python_type()("2018")), datetime)
        dtp = DateDataType(format_string="%Y")
        self.assertEqual(type(dtp.get_python_type()("2018")), datetime)

    def test_build_numpy_value(self):
        dtp = DateDataType(resolution='Y', format_string="%Y-%m-%d")
        self.assertEqual(dtp.build_numpy_value("2018-04-01"), np.datetime64("2018"))
        dtp = DateDataType(resolution='M', format_string="%Y-%m-%d")
        self.assertEqual(dtp.build_numpy_value("2018-04-01"), np.datetime64("2018-04"))
        dtp = DateDataType(format_string="%Y-%m-%d")
        self.assertEqual(dtp.build_numpy_value("2018-04-01"), np.datetime64("2018-04-01"))

    def test_build_python_value(self):
        dtp = DateDataType(resolution='Y', format_string="%Y-%m-%d")
        self.assertEqual(dtp.build_python_value("2018-04-01"), datetime.strptime("2018-04-01", "%Y-%m-%d"))
        dtp = DateDataType(resolution='M', format_string="%Y-%m")
        self.assertEqual(dtp.build_python_value("2018-04"), datetime.strptime("2018-04", "%Y-%m"))
        dtp = DateDataType(format_string="%Y")
        self.assertEqual(dtp.build_python_value("2018"), datetime.strptime("2018", "%Y"))
