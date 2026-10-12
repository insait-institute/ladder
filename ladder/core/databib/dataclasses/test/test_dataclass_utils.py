import unittest

from ladder.core.databib.dataclasses import Dataclass, DataclassDict, dataclass, zip_dataclass_items


@dataclass
class TestDataClass(Dataclass):
    a: str
    b: int


@dataclass
class TestNestedDataClass(Dataclass):
    c: TestDataClass
    d: TestDataClass
    e: float


@dataclass
class TestMixedDataClass(Dataclass):
    f: DataclassDict[TestDataClass]
    g: TestDataClass
    h: float


class DataclassUtilsTest(unittest.TestCase):
    def test_zip_dataclass_items(self):
        x = TestNestedDataClass(
            c=TestDataClass(a='A', b=5),
            d=TestDataClass(a='A', b=6),
            e=0.7,
        )
        y = TestNestedDataClass(
            c=TestDataClass(a='A1', b=7),
            d=TestDataClass(a='A1', b=8),
            e=0.9,
        )

        self.assertDictEqual(
            {
                "c": (
                    TestDataClass(a='A', b=5),
                    TestDataClass(a='A1', b=7),
                ),
                "d": (
                    TestDataClass(a='A', b=6),
                    TestDataClass(a='A1', b=8),
                ),
                "e": (0.7, 0.9),
            },
            {key: value for key, value in zip_dataclass_items(x, y, recursive=False)},
        )

        self.assertDictEqual(
            {
                "c.a": ("A", "A1"),
                "c.b": (5, 7),
                "d.a": ("A", "A1"),
                "d.b": (6, 8),
                "e": (0.7, 0.9),
            },
            {key: value for key, value in zip_dataclass_items(x, y, recursive=True)},
        )

        x = TestMixedDataClass(
            f=DataclassDict[TestDataClass](i=TestDataClass(a='A', b=5), j=TestDataClass(a='A', b=5)),
            g=TestDataClass(a='A', b=6),
            h=0.7,
        )
        y = TestMixedDataClass(
            f=DataclassDict[TestDataClass](i=TestDataClass(a='A1', b=7)),
            g=TestDataClass(a='A1', b=8),
            h=0.9,
        )

        with self.assertRaises(ValueError):
            {  # pylint:disable=expression-not-assigned
                key: value for key, value in zip_dataclass_items(x, y, recursive=True)
            }


if __name__ == '__main__':
    unittest.main()
