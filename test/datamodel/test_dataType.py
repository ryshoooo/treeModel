from unittest import TestCase
import numpy as np
from datetime import datetime

from treemodel.datamodel.datatypes import DataType, StringDataType, FloatDataType, DateDataType, ArrayDataType, ListDataType


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

    def test_eq(self):
        dtp1 = DataType(numpy_dtype='<i4', python_dtype=int, numpy_na_value=None, python_na_value=None)
        dtp2 = DataType(numpy_dtype='<i4', python_dtype=int, numpy_na_value=None, python_na_value=None)
        self.assertEqual(dtp1, dtp2)
        dtp1 = DataType(numpy_dtype='<c16', python_dtype=float, numpy_na_value=None, python_na_value=None)
        dtp2 = DataType(numpy_dtype='<c16', python_dtype=float, numpy_na_value=None, python_na_value=None)
        self.assertEqual(dtp1, dtp2)
        dtp1 = DataType(numpy_dtype='<a25', python_dtype=bytes, numpy_na_value=None, python_na_value=None)
        dtp2 = DataType(numpy_dtype='<a25', python_dtype=bytes, numpy_na_value=None, python_na_value=None)
        self.assertEqual(dtp1, dtp2)


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
        dtp = StringDataType()
        self.assertEqual(dtp.get_numpy_type(), np.dtype('<U128'))

    def test_get_python_type(self):
        dtp = StringDataType()
        self.assertEqual(dtp.get_python_type(), str)

    def test_build_numpy_value(self):
        dtp = StringDataType()
        self.assertEqual(dtp.build_numpy_value("1234567890123"), "1234567890123")
        self.assertEqual(dtp.build_numpy_value("123"), "123")
        self.assertEqual(dtp.build_numpy_value("tra2"), "tra2")

    def test_build_python_value(self):
        dtp = StringDataType()
        self.assertEqual(dtp.build_python_value(10), "10")
        self.assertEqual(dtp.build_python_value(10), "10")
        self.assertEqual(dtp.build_python_value("tra2"), "tra2")

    def test_eq(self):
        dtp1 = StringDataType()
        dtp2 = StringDataType()
        self.assertEqual(dtp1, dtp2)


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

    def test_eq(self):
        dtp1 = FloatDataType(bits=2)
        dtp2 = FloatDataType(bits=2)
        self.assertEqual(dtp1, dtp2)
        dtp1 = FloatDataType(bits=2)
        dtp2 = FloatDataType(bits=3)
        self.assertEqual(dtp1, dtp2)


class TestDateDataType(TestCase):
    """
    Test class for StringDataType.
    """

    def test__datetime_format(self):
        self.assertEqual(DateDataType._datetime_format("2018", "%Y"), datetime(2018, 1, 1))
        self.assertEqual(DateDataType._datetime_format("2018-03", "%Y-%m"), datetime(2018, 3, 1))
        self.assertEqual(DateDataType._datetime_format("2018-03-29", "%Y-%m-%d"), datetime(2018, 3, 29))
        self.assertEqual(DateDataType._datetime_format("2018-03-29 18", "%Y-%m-%d %H"), datetime(2018, 3, 29, 18))
        self.assertEqual(DateDataType._datetime_format("2018-03-29 18:36", "%Y-%m-%d %H:%M"),
                         datetime(2018, 3, 29, 18, 36))
        self.assertEqual(DateDataType._datetime_format("2018-03-29 18:36:59", "%Y-%m-%d %H:%M:%S"),
                         datetime(2018, 3, 29, 18, 36, 59))
        self.assertEqual(DateDataType._datetime_format("2018-03-29 18:36:59.967344", "%Y-%m-%d %H:%M:%S.%f"),
                         datetime(2018, 3, 29, 18, 36, 59, 967344))

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

    def test_eq(self):
        dtp1 = DateDataType(resolution='Y', format_string="%Y-%m-%d")
        dtp2 = DateDataType(resolution='Y', format_string="%Y-%m-%d")
        self.assertEqual(dtp1, dtp2)
        dtp1 = DateDataType(resolution='Y', format_string="%Y-%m-%d %H")
        dtp2 = DateDataType(resolution='Y', format_string="%Y-%m-%d")
        self.assertEqual(dtp1, dtp2)
        dtp1 = DateDataType(resolution='D', format_string="%Y-%m-%d %H")
        dtp2 = DateDataType(resolution='D', format_string="%Y-%m-%d")
        self.assertEqual(dtp1, dtp2)
        dtp1 = DateDataType(resolution='D', format_string="%Y-%m-%d %H")
        dtp2 = DateDataType(resolution='Y', format_string="%Y-%m-%d")
        self.assertNotEqual(dtp1, dtp2)


class TestArrayDataType(TestCase):
    """
    Test class for ArrayDataType.
    """

    def test_is_nullable(self):
        dtp = ArrayDataType(element_data_type=StringDataType(), nullable=False)
        self.assertFalse(dtp.is_nullable())
        dtp = ArrayDataType(element_data_type=StringDataType(), nullable=True)
        self.assertTrue(dtp.is_nullable())

    def test_get_numpy_type(self):
        dtp = ArrayDataType(element_data_type=FloatDataType())
        self.assertEqual(dtp.get_numpy_type(), np.ndarray)
        dtp = ArrayDataType(element_data_type=StringDataType())
        self.assertEqual(dtp.get_numpy_type(), np.ndarray)
        dtp = ArrayDataType(element_data_type=DateDataType())
        self.assertEqual(dtp.get_numpy_type(), np.ndarray)

    def test_get_python_type(self):
        dtp = ArrayDataType(element_data_type=FloatDataType())
        self.assertEqual(dtp.get_python_type(), list)
        dtp = ArrayDataType(element_data_type=StringDataType())
        self.assertEqual(dtp.get_python_type(), list)
        dtp = ArrayDataType(element_data_type=DateDataType())
        self.assertEqual(dtp.get_python_type(), list)

    def test_build_numpy_value(self):
        dtp = ArrayDataType(element_data_type=FloatDataType())
        self.assertTrue((dtp.build_numpy_value([1, 2, 3]) == np.array([1, 2, 3], '<f8')).all())
        dtp = ArrayDataType(element_data_type=StringDataType())
        self.assertTrue((dtp.build_numpy_value([1, 2, 3]) == np.array([1, 2, 3], '<U200')).all())
        dtp = ArrayDataType(element_data_type=ArrayDataType(element_data_type=StringDataType()))
        self.assertTrue(
            (dtp.build_numpy_value([["tra", "check"], ["what"]])[0] == np.array(["tra", "check"], '<U200')).all())
        self.assertTrue(
            (dtp.build_numpy_value([["tra", "check"], ["what"]])[1] == np.array(["what"], "<U200")).all())

    def test_build_python_value(self):
        dtp = ArrayDataType(element_data_type=FloatDataType())
        self.assertTrue((dtp.build_python_value([1, 2, 3]) == np.array([1, 2, 3], '<f8')).all())
        dtp = ArrayDataType(element_data_type=StringDataType())
        self.assertTrue((dtp.build_python_value([1, 2, 3]) == np.array([1, 2, 3], '<U200')).all())
        dtp = ArrayDataType(element_data_type=ArrayDataType(element_data_type=StringDataType()))
        self.assertTrue(
            (dtp.build_python_value([["tra", "check"], ["what"]])[0] == np.array(["tra", "check"], '<U200')).all())
        self.assertTrue(
            (dtp.build_python_value([["tra", "check"], ["what"]])[1] == np.array(["what"], "<U200")).all())

    def test_eq(self):
        dtp1 = ArrayDataType(element_data_type=FloatDataType())
        dtp2 = ArrayDataType(element_data_type=FloatDataType())
        self.assertEqual(dtp1, dtp2)
        dtp1 = ArrayDataType(element_data_type=StringDataType())
        dtp2 = ArrayDataType(element_data_type=FloatDataType())
        self.assertNotEqual(dtp1, dtp2)


class TestListDataType(TestCase):
    """
    Test class for ListDataType.
    """

    def test_is_nullable(self):
        dtp = ListDataType(element_data_types=[StringDataType()], nullable=False)
        self.assertFalse(dtp.is_nullable())
        dtp = ListDataType(element_data_types=[StringDataType()], nullable=True)
        self.assertTrue(dtp.is_nullable())

    def test_get_numpy_type(self):
        dtp = ListDataType(element_data_types=[FloatDataType()])
        self.assertEqual(dtp.get_numpy_type(), np.ndarray)
        dtp = ListDataType(element_data_types=[StringDataType()])
        self.assertEqual(dtp.get_numpy_type(), np.ndarray)
        dtp = ListDataType(element_data_types=[DateDataType()])
        self.assertEqual(dtp.get_numpy_type(), np.ndarray)

    def test_get_python_type(self):
        dtp = ListDataType(element_data_types=[FloatDataType()])
        self.assertEqual(dtp.get_python_type(), list)
        dtp = ListDataType(element_data_types=[StringDataType()])
        self.assertEqual(dtp.get_python_type(), list)
        dtp = ListDataType(element_data_types=[DateDataType()])
        self.assertEqual(dtp.get_python_type(), list)

    def test__get_numpy_dtypes(self):
        dtp = ListDataType(element_data_types=[FloatDataType()])
        self.assertEqual(dtp._get_numpy_dtypes(), [('0', '<f8')])
        dtp = ListDataType(element_data_types=[FloatDataType(), ArrayDataType(element_data_type=StringDataType())])
        self.assertEqual(dtp._get_numpy_dtypes(), [('0', '<f8'), ('1', np.ndarray)])
        dtp = ListDataType(element_data_types=[FloatDataType(), ArrayDataType(element_data_type=StringDataType()),
                                               DateDataType(resolution='M')])
        self.assertEqual(dtp._get_numpy_dtypes(), [('0', '<f8'), ('1', np.ndarray), ('2', '<M8[M]')])

    def test_build_numpy_value(self):
        dtp = ListDataType(element_data_types=[FloatDataType()])
        self.assertTrue((dtp.build_numpy_value([1]) == np.array((1,), [('0', '<f8')])).all())
        dtp = ListDataType(element_data_types=[FloatDataType(), StringDataType()])
        self.assertTrue(
            (dtp.build_numpy_value([1, "tra"]) == np.array((1, "tra"), [('0', '<f8'), ('1', '<U128')])).all())

        dtp = ListDataType(
            element_data_types=[
                FloatDataType(),
                StringDataType(),
                ListDataType(
                    element_data_types=[
                        ArrayDataType(element_data_type=FloatDataType()),
                        StringDataType()
                    ]
                )
            ]
        )

        input_value = [12.3, "first_string", [[1, 2, 3, 4], "second_string"]]
        output_value = dtp.build_numpy_value(input_value)

        self.assertEqual(output_value[0]['0'], input_value[0])
        self.assertEqual(output_value[0]['1'], input_value[1])
        self.assertTrue((output_value[0]['2'][0]['0'] == input_value[2][0]).all())
        self.assertEqual(output_value[0]['2'][0]['1'], input_value[2][1])

    def test_build_python_value(self):
        dtp = ListDataType(element_data_types=[FloatDataType()])
        self.assertTrue((dtp.build_python_value([1]) == [float(1)]))
        dtp = ListDataType(element_data_types=[FloatDataType(), StringDataType()])
        self.assertTrue(dtp.build_python_value([1, "tra"]) == [float(1), "tra"])

        dtp = ListDataType(
            element_data_types=[
                FloatDataType(),
                StringDataType(),
                ListDataType(
                    element_data_types=[
                        ArrayDataType(element_data_type=FloatDataType()),
                        StringDataType()
                    ]
                )
            ]
        )

        input_value = [12.3, "first_string", [[1, 2, 3, 4], "second_string"]]
        output_value = dtp.build_python_value(input_value)

        self.assertEqual(output_value[0], input_value[0])
        self.assertEqual(output_value[1], input_value[1])
        self.assertTrue((output_value[2][0] == input_value[2][0]))
        self.assertEqual(output_value[2][1], input_value[2][1])

    def test_eq(self):
        dtp1 = ListDataType(element_data_types=[FloatDataType()])
        dtp2 = ListDataType(element_data_types=[FloatDataType()])
        self.assertEqual(dtp1, dtp2)
        dtp1 = ListDataType(element_data_types=[FloatDataType(), StringDataType()])
        dtp2 = ListDataType(element_data_types=[FloatDataType(), StringDataType()])
        self.assertEqual(dtp1, dtp2)
        dtp1 = ListDataType(
            element_data_types=[
                FloatDataType(),
                StringDataType(),
                ListDataType(
                    element_data_types=[
                        ArrayDataType(element_data_type=FloatDataType()),
                        StringDataType()
                    ]
                )
            ]
        )
        dtp2 = ListDataType(
            element_data_types=[
                FloatDataType(),
                StringDataType(),
                ListDataType(
                    element_data_types=[
                        ArrayDataType(element_data_type=FloatDataType()),
                        StringDataType()
                    ]
                )
            ]
        )
        self.assertEqual(dtp1, dtp2)


class TestComparisons(TestCase):
    """
    Test class for comparisons
    """

    @staticmethod
    def get_data_types():
        dt = DataType(numpy_dtype='<i8', python_dtype=int, numpy_na_value=np.nan, python_na_value=None)
        sdt = StringDataType()
        fdt = FloatDataType()
        ddt_d = DateDataType(resolution='D')
        ddt_s = DateDataType(resolution='s')
        adt_f = ArrayDataType(element_data_type=FloatDataType())
        adt_s = ArrayDataType(element_data_type=StringDataType())
        ldt_fsd = ListDataType(element_data_types=[FloatDataType(), StringDataType(), DateDataType()])
        ldt_ssd = ListDataType(element_data_types=[StringDataType(), StringDataType(), DateDataType()])

        return dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd

    def test_comparisons_data_type(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd = self.get_data_types()
        self.assertTrue(dt <= dt)
        self.assertFalse(fdt <= dt)
        self.assertFalse(ddt_d <= dt)
        self.assertFalse(ddt_s <= dt)
        self.assertFalse(adt_f <= dt)
        self.assertFalse(adt_s <= dt)
        self.assertFalse(ldt_fsd <= dt)
        self.assertFalse(ldt_ssd <= dt)
        self.assertFalse(sdt <= dt)

        self.assertFalse(dt > dt)
        self.assertFalse(fdt > dt)
        self.assertFalse(ddt_d > dt)
        self.assertFalse(ddt_s > dt)
        self.assertFalse(adt_f > dt)
        self.assertFalse(adt_s > dt)
        self.assertFalse(ldt_fsd > dt)
        self.assertFalse(ldt_ssd > dt)
        self.assertTrue(sdt > dt)

        self.assertTrue(dt >= dt)
        self.assertFalse(dt >= fdt)
        self.assertFalse(dt >= ddt_d)
        self.assertFalse(dt >= ddt_s)
        self.assertFalse(dt >= adt_f)
        self.assertFalse(dt >= adt_s)
        self.assertFalse(dt >= ldt_fsd)
        self.assertFalse(dt >= ldt_ssd)
        self.assertFalse(dt >= sdt)

        self.assertFalse(dt < dt)
        self.assertFalse(dt < fdt)
        self.assertFalse(dt < ddt_d)
        self.assertFalse(dt < ddt_s)
        self.assertFalse(dt < adt_f)
        self.assertFalse(dt < adt_s)
        self.assertFalse(dt < ldt_fsd)
        self.assertFalse(dt < ldt_ssd)
        self.assertTrue(dt < sdt)

    def test_comparisons_string(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd = self.get_data_types()

        self.assertTrue(dt <= sdt)
        self.assertTrue(fdt <= sdt)
        self.assertTrue(ddt_d <= sdt)
        self.assertTrue(ddt_s <= sdt)
        self.assertTrue(adt_f <= sdt)
        self.assertTrue(adt_s <= sdt)
        self.assertTrue(ldt_fsd <= sdt)
        self.assertTrue(ldt_ssd <= sdt)
        self.assertTrue(sdt <= sdt)

        self.assertFalse(dt > sdt)
        self.assertFalse(fdt > sdt)
        self.assertFalse(ddt_d > sdt)
        self.assertFalse(ddt_s > sdt)
        self.assertFalse(adt_f > sdt)
        self.assertFalse(adt_s > sdt)
        self.assertFalse(ldt_fsd > sdt)
        self.assertFalse(ldt_ssd > sdt)
        self.assertFalse(sdt > sdt)

        self.assertTrue(sdt >= dt)
        self.assertTrue(sdt >= fdt)
        self.assertTrue(sdt >= ddt_d)
        self.assertTrue(sdt >= ddt_s)
        self.assertTrue(sdt >= adt_f)
        self.assertTrue(sdt >= adt_s)
        self.assertTrue(sdt >= ldt_fsd)
        self.assertTrue(sdt >= ldt_ssd)
        self.assertTrue(sdt >= sdt)

        self.assertFalse(sdt < dt)
        self.assertFalse(sdt < fdt)
        self.assertFalse(sdt < ddt_d)
        self.assertFalse(sdt < ddt_s)
        self.assertFalse(sdt < adt_f)
        self.assertFalse(sdt < adt_s)
        self.assertFalse(sdt < ldt_fsd)
        self.assertFalse(sdt < ldt_ssd)
        self.assertFalse(sdt < sdt)

    def test_comparisons_float(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd = self.get_data_types()

        self.assertFalse(dt <= fdt)
        self.assertTrue(fdt <= fdt)
        self.assertFalse(ddt_d <= fdt)
        self.assertFalse(ddt_s <= fdt)
        self.assertFalse(adt_f <= fdt)
        self.assertFalse(adt_s <= fdt)
        self.assertFalse(ldt_fsd <= fdt)
        self.assertFalse(ldt_ssd <= fdt)
        self.assertFalse(sdt <= fdt)

        self.assertFalse(dt > fdt)
        self.assertFalse(fdt > fdt)
        self.assertFalse(ddt_d > fdt)
        self.assertFalse(ddt_s > fdt)
        self.assertFalse(adt_f > fdt)
        self.assertFalse(adt_s > fdt)
        self.assertFalse(ldt_fsd > fdt)
        self.assertFalse(ldt_ssd > fdt)
        self.assertTrue(sdt > fdt)

        self.assertFalse(fdt >= dt)
        self.assertTrue(fdt >= fdt)
        self.assertFalse(fdt >= ddt_d)
        self.assertFalse(fdt >= ddt_s)
        self.assertFalse(fdt >= adt_f)
        self.assertFalse(fdt >= adt_s)
        self.assertFalse(fdt >= ldt_fsd)
        self.assertFalse(fdt >= ldt_ssd)
        self.assertFalse(fdt >= sdt)

        self.assertFalse(fdt < dt)
        self.assertFalse(fdt < fdt)
        self.assertFalse(fdt < ddt_d)
        self.assertFalse(fdt < ddt_s)
        self.assertFalse(fdt < adt_f)
        self.assertFalse(fdt < adt_s)
        self.assertFalse(fdt < ldt_fsd)
        self.assertFalse(fdt < ldt_ssd)
        self.assertTrue(fdt < sdt)

    def test_comparisons_date_1(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd = self.get_data_types()

        self.assertFalse(dt <= ddt_d)
        self.assertFalse(fdt <= ddt_d)
        self.assertTrue(ddt_d <= ddt_d)
        self.assertFalse(ddt_s <= ddt_d)
        self.assertFalse(adt_f <= ddt_d)
        self.assertFalse(adt_s <= ddt_d)
        self.assertFalse(ldt_fsd <= ddt_d)
        self.assertFalse(ldt_ssd <= ddt_d)
        self.assertFalse(sdt <= ddt_d)

        self.assertFalse(dt > ddt_d)
        self.assertFalse(fdt > ddt_d)
        self.assertFalse(ddt_d > ddt_d)
        self.assertTrue(ddt_s > ddt_d)
        self.assertFalse(adt_f > ddt_d)
        self.assertFalse(adt_s > ddt_d)
        self.assertFalse(ldt_fsd > ddt_d)
        self.assertFalse(ldt_ssd > ddt_d)
        self.assertTrue(sdt > ddt_d)

        self.assertFalse(ddt_d >= dt)
        self.assertFalse(ddt_d >= fdt)
        self.assertTrue(ddt_d >= ddt_d)
        self.assertFalse(ddt_d >= ddt_s)
        self.assertFalse(ddt_d >= adt_f)
        self.assertFalse(ddt_d >= adt_s)
        self.assertFalse(ddt_d >= ldt_fsd)
        self.assertFalse(ddt_d >= ldt_ssd)
        self.assertFalse(ddt_d >= sdt)

        self.assertFalse(ddt_d < dt)
        self.assertFalse(ddt_d < fdt)
        self.assertFalse(ddt_d < ddt_d)
        self.assertTrue(ddt_d < ddt_s)
        self.assertFalse(ddt_d < adt_f)
        self.assertFalse(ddt_d < adt_s)
        self.assertFalse(ddt_d < ldt_fsd)
        self.assertFalse(ddt_d < ldt_ssd)
        self.assertTrue(ddt_d < sdt)

    def test_comparisons_date_2(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd = self.get_data_types()

        self.assertFalse(dt <= ddt_s)
        self.assertFalse(fdt <= ddt_s)
        self.assertTrue(ddt_d <= ddt_s)
        self.assertTrue(ddt_s <= ddt_s)
        self.assertFalse(adt_f <= ddt_s)
        self.assertFalse(adt_s <= ddt_s)
        self.assertFalse(ldt_fsd <= ddt_s)
        self.assertFalse(ldt_ssd <= ddt_s)
        self.assertFalse(sdt <= ddt_s)

        self.assertFalse(dt > ddt_s)
        self.assertFalse(fdt > ddt_s)
        self.assertFalse(ddt_d > ddt_s)
        self.assertFalse(ddt_s > ddt_s)
        self.assertFalse(adt_f > ddt_s)
        self.assertFalse(adt_s > ddt_s)
        self.assertFalse(ldt_fsd > ddt_s)
        self.assertFalse(ldt_ssd > ddt_s)
        self.assertTrue(sdt > ddt_s)

        self.assertFalse(ddt_s >= dt)
        self.assertFalse(ddt_s >= fdt)
        self.assertTrue(ddt_s >= ddt_d)
        self.assertTrue(ddt_s >= ddt_s)
        self.assertFalse(ddt_s >= adt_f)
        self.assertFalse(ddt_s >= adt_s)
        self.assertFalse(ddt_s >= ldt_fsd)
        self.assertFalse(ddt_s >= ldt_ssd)
        self.assertFalse(ddt_s >= sdt)

        self.assertFalse(ddt_s < dt)
        self.assertFalse(ddt_s < fdt)
        self.assertFalse(ddt_s < ddt_d)
        self.assertFalse(ddt_s < ddt_s)
        self.assertFalse(ddt_s < adt_f)
        self.assertFalse(ddt_s < adt_s)
        self.assertFalse(ddt_s < ldt_fsd)
        self.assertFalse(ddt_s < ldt_ssd)
        self.assertTrue(ddt_s < sdt)

    def test_comparisons_array_1(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd = self.get_data_types()

        self.assertFalse(dt <= adt_f)
        self.assertFalse(fdt <= adt_f)
        self.assertFalse(ddt_d <= adt_f)
        self.assertFalse(ddt_s <= adt_f)
        self.assertTrue(adt_f <= adt_f)
        self.assertFalse(adt_s <= adt_f)
        self.assertFalse(ldt_fsd <= adt_f)
        self.assertFalse(ldt_ssd <= adt_f)
        self.assertFalse(sdt <= adt_f)

        self.assertFalse(dt > adt_f)
        self.assertFalse(fdt > adt_f)
        self.assertFalse(ddt_d > adt_f)
        self.assertFalse(ddt_s > adt_f)
        self.assertFalse(adt_f > adt_f)
        self.assertTrue(adt_s > adt_f)
        self.assertTrue(ldt_fsd > adt_f)
        self.assertTrue(ldt_ssd > adt_f)
        self.assertTrue(sdt > adt_f)

        self.assertFalse(adt_f >= dt)
        self.assertFalse(adt_f >= fdt)
        self.assertFalse(adt_f >= ddt_d)
        self.assertFalse(adt_f >= ddt_s)
        self.assertTrue(adt_f >= adt_f)
        self.assertFalse(adt_f >= adt_s)
        self.assertFalse(adt_f >= ldt_fsd)
        self.assertFalse(adt_f >= ldt_ssd)
        self.assertFalse(adt_f >= sdt)

        self.assertFalse(adt_f < dt)
        self.assertFalse(adt_f < fdt)
        self.assertFalse(adt_f < ddt_d)
        self.assertFalse(adt_f < ddt_s)
        self.assertFalse(adt_f < adt_f)
        self.assertTrue(adt_f < adt_s)
        self.assertTrue(adt_f < ldt_fsd)
        self.assertTrue(adt_f < ldt_ssd)
        self.assertTrue(adt_f < sdt)

    def test_comparisons_array_2(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd = self.get_data_types()

        self.assertFalse(dt <= adt_s)
        self.assertFalse(fdt <= adt_s)
        self.assertFalse(ddt_d <= adt_s)
        self.assertFalse(ddt_s <= adt_s)
        self.assertTrue(adt_f <= adt_s)
        self.assertTrue(adt_s <= adt_s)
        self.assertFalse(ldt_fsd <= adt_s)
        self.assertFalse(ldt_ssd <= adt_s)
        self.assertFalse(sdt <= adt_s)

        self.assertFalse(dt > adt_s)
        self.assertFalse(fdt > adt_s)
        self.assertFalse(ddt_d > adt_s)
        self.assertFalse(ddt_s > adt_s)
        self.assertFalse(adt_f > adt_s)
        self.assertFalse(adt_s > adt_s)
        self.assertTrue(ldt_fsd > adt_s)
        self.assertTrue(ldt_ssd > adt_s)
        self.assertTrue(sdt > adt_s)

        self.assertFalse(adt_s >= dt)
        self.assertFalse(adt_s >= fdt)
        self.assertFalse(adt_s >= ddt_d)
        self.assertFalse(adt_s >= ddt_s)
        self.assertTrue(adt_s >= adt_f)
        self.assertTrue(adt_s >= adt_s)
        self.assertFalse(adt_s >= ldt_fsd)
        self.assertFalse(adt_s >= ldt_ssd)
        self.assertFalse(adt_s >= sdt)

        self.assertFalse(adt_s < dt)
        self.assertFalse(adt_s < fdt)
        self.assertFalse(adt_s < ddt_d)
        self.assertFalse(adt_s < ddt_s)
        self.assertFalse(adt_s < adt_f)
        self.assertFalse(adt_s < adt_s)
        self.assertTrue(adt_s < ldt_fsd)
        self.assertTrue(adt_s < ldt_ssd)
        self.assertTrue(adt_s < sdt)

    def test_comparisons_list_1(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd = self.get_data_types()

        self.assertFalse(dt <= ldt_fsd)
        self.assertFalse(fdt <= ldt_fsd)
        self.assertFalse(ddt_d <= ldt_fsd)
        self.assertFalse(ddt_s <= ldt_fsd)
        self.assertTrue(adt_f <= ldt_fsd)
        self.assertTrue(adt_s <= ldt_fsd)
        self.assertTrue(ldt_fsd <= ldt_fsd)
        self.assertFalse(ldt_ssd <= ldt_fsd)
        self.assertFalse(sdt <= ldt_fsd)

        self.assertFalse(dt > ldt_fsd)
        self.assertFalse(fdt > ldt_fsd)
        self.assertFalse(ddt_d > ldt_fsd)
        self.assertFalse(ddt_s > ldt_fsd)
        self.assertFalse(adt_f > ldt_fsd)
        self.assertFalse(adt_s > ldt_fsd)
        self.assertFalse(ldt_fsd > ldt_fsd)
        self.assertFalse(ldt_ssd > ldt_fsd)
        self.assertTrue(sdt > ldt_fsd)

        self.assertFalse(ldt_fsd >= dt)
        self.assertFalse(ldt_fsd >= fdt)
        self.assertFalse(ldt_fsd >= ddt_d)
        self.assertFalse(ldt_fsd >= ddt_s)
        self.assertTrue(ldt_fsd >= adt_f)
        self.assertTrue(ldt_fsd >= adt_s)
        self.assertTrue(ldt_fsd >= ldt_fsd)
        self.assertFalse(ldt_fsd >= ldt_ssd)
        self.assertFalse(ldt_fsd >= sdt)

        self.assertFalse(ldt_fsd < dt)
        self.assertFalse(ldt_fsd < fdt)
        self.assertFalse(ldt_fsd < ddt_d)
        self.assertFalse(ldt_fsd < ddt_s)
        self.assertFalse(ldt_fsd < adt_f)
        self.assertFalse(ldt_fsd < adt_s)
        self.assertFalse(ldt_fsd < ldt_fsd)
        self.assertFalse(ldt_fsd < ldt_ssd)
        self.assertTrue(ldt_fsd < sdt)

    def test_comparisons_list_2(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd = self.get_data_types()

        self.assertFalse(dt <= ldt_ssd)
        self.assertFalse(fdt <= ldt_ssd)
        self.assertFalse(ddt_d <= ldt_ssd)
        self.assertFalse(ddt_s <= ldt_ssd)
        self.assertTrue(adt_f <= ldt_ssd)
        self.assertTrue(adt_s <= ldt_ssd)
        self.assertTrue(ldt_fsd <= ldt_ssd)
        self.assertTrue(ldt_ssd <= ldt_ssd)
        self.assertFalse(sdt <= ldt_ssd)

        self.assertFalse(dt > ldt_ssd)
        self.assertFalse(fdt > ldt_ssd)
        self.assertFalse(ddt_d > ldt_ssd)
        self.assertFalse(ddt_s > ldt_ssd)
        self.assertFalse(adt_f > ldt_ssd)
        self.assertFalse(adt_s > ldt_ssd)
        self.assertFalse(ldt_fsd > ldt_ssd)
        self.assertFalse(ldt_ssd > ldt_ssd)
        self.assertTrue(sdt > ldt_ssd)

        self.assertFalse(ldt_ssd >= dt)
        self.assertFalse(ldt_ssd >= fdt)
        self.assertFalse(ldt_ssd >= ddt_d)
        self.assertFalse(ldt_ssd >= ddt_s)
        self.assertTrue(ldt_ssd >= adt_f)
        self.assertTrue(ldt_ssd >= adt_s)
        self.assertTrue(ldt_ssd >= ldt_fsd)
        self.assertTrue(ldt_ssd >= ldt_ssd)
        self.assertFalse(ldt_ssd >= sdt)

        self.assertFalse(ldt_ssd < dt)
        self.assertFalse(ldt_ssd < fdt)
        self.assertFalse(ldt_ssd < ddt_d)
        self.assertFalse(ldt_ssd < ddt_s)
        self.assertFalse(ldt_ssd < adt_f)
        self.assertFalse(ldt_ssd < adt_s)
        self.assertFalse(ldt_ssd < ldt_fsd)
        self.assertFalse(ldt_ssd < ldt_ssd)
        self.assertTrue(ldt_ssd < sdt)
