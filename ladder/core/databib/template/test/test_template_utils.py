from typing import TypeVar

from ladder.core.databib.template.template_utils import resolve_type_var_attrs_chain

T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')


class TestTemplateUtils:
    def test_resolve_type_var_attrs_chain(self):
        self._test_resolve_type_var_attrs_chain(
            type_var_attrs={T1: T1},
            expected_result={T1: T1},
        )

        self._test_resolve_type_var_attrs_chain(
            type_var_attrs={T1: T2, T2: T2},
            expected_result={T1: T2, T2: T2},
        )

        self._test_resolve_type_var_attrs_chain(
            type_var_attrs={T1: T2, T2: T3, T3: T3},
            expected_result={T1: T3, T2: T3, T3: T3},
        )

        self._test_resolve_type_var_attrs_chain(
            type_var_attrs={T1: T2, T2: T3, T3: int},
            expected_result={T1: int, T2: int, T3: int},
        )

    def _test_resolve_type_var_attrs_chain(self, type_var_attrs, expected_result):
        actual_result = resolve_type_var_attrs_chain(type_var_attrs)
        assert expected_result == actual_result
