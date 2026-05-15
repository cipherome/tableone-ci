import warnings
from math import isclose

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tableone import TableOne
from tableone.tableone import InputError


def mytest(*args):
    mytest.__name__ = "Test name"
    _, pval = stats.ks_2samp(*args)
    return pval


class TestPValues(object):
    def test_pval_correction_missing_value_on_separate_row(self):
        df = pd.DataFrame(
            {
                "numbers": [1, 2, 6, 1, 1, 1],
                "other": [1, 2, 3, 3, 3, 4],
                "colors": ["red", "white", "blue", "red", "blue", "blue"],
                "even": ["yes", "no", "yes", "yes", "no", "yes"],
            }
        )

        t1 = TableOne(df, groupby="even", pval=True, pval_adjust="bonferroni", missing_value_on_separate_row=True)

        pvals_expected = {"numbers, mean (SD)": "1.000", "other, mean (SD)": "1.000", "colors, n (%)": "0.669"}

        group = "Grouped by even"
        col = "P-Value (adjusted)"
        for k in pvals_expected:
            assert t1.tableone.loc[k][group][col].values[0] == pvals_expected[k]

        with warnings.catch_warnings(record=False):
            warnings.simplefilter("ignore", category=UserWarning)
            TableOne(df, groupby="even", pval=True, pval_adjust=True)

        for k in pvals_expected:
            assert t1.tableone.loc[k][group][col].values[0] == pvals_expected[k]

    def test_pval_correction(self):
        df = pd.DataFrame(
            {
                "numbers": [1, 2, 6, 1, 1, 1],
                "other": [1, 2, 3, 3, 3, 4],
                "colors": ["red", "white", "blue", "red", "blue", "blue"],
                "even": ["yes", "no", "yes", "yes", "no", "yes"],
            }
        )

        t1 = TableOne(df, groupby="even", pval=True, pval_adjust="bonferroni", missing_value_on_separate_row=False)

        pvals_expected = {"numbers, mean (SD)": "1.000", "other, mean (SD)": "1.000", "colors, n (%)": "0.669"}

        group = "Grouped by even"
        col = "P-Value (adjusted)"
        for k in pvals_expected:
            assert t1.tableone.loc[k][group][col].values[0] == pvals_expected[k]

        with warnings.catch_warnings(record=False):
            warnings.simplefilter("ignore", category=UserWarning)
            TableOne(df, groupby="even", pval=True, pval_adjust=True)

        for k in pvals_expected:
            assert t1.tableone.loc[k][group][col].values[0] == pvals_expected[k]

    def test_pval_requires_groupby(self):
        df = pd.DataFrame({"numbers": [1, 2, 3, 4]})

        with pytest.raises(InputError):
            TableOne(df, columns=["numbers"], pval=True)

    def test_pval_adjust_invalid_method_raises_error(self):
        df = pd.DataFrame({"group": ["a", "a", "b", "b"], "value": [1, 2, 3, 4]})

        with pytest.raises(ValueError):
            TableOne(df, columns=["value"], groupby="group", pval=True, pval_adjust="not-a-method")

    def test_welch_ttest_for_two_normal_groups(self):
        df = pd.DataFrame({"group": ["a"] * 4 + ["b"] * 4, "value": [1, 2, 3, 4, 4, 5, 6, 7]})

        table = TableOne(df, columns=["value"], groupby="group", pval=True)
        expected = stats.ttest_ind([1, 2, 3, 4], [4, 5, 6, 7], equal_var=False, nan_policy="omit").pvalue

        assert table._htest_table.loc["value", "Test"] == "Two Sample T-test"
        assert table._htest_table.loc["value", "P-Value"] == pytest.approx(expected)

    def test_anova_for_three_normal_groups(self):
        df = pd.DataFrame(
            {
                "group": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
                "value": [1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11],
            }
        )

        table = TableOne(df, columns=["value"], groupby="group", pval=True)
        expected = stats.f_oneway([1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]).pvalue

        assert table._htest_table.loc["value", "Test"] == "One-way ANOVA"
        assert table._htest_table.loc["value", "P-Value"] == pytest.approx(expected)

    def test_kruskal_for_nonnormal_groups(self):
        df = pd.DataFrame(
            {
                "group": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
                "value": [1, 1, 2, 3, 2, 4, 8, 16, 3, 9, 27, 81],
            }
        )

        table = TableOne(df, columns=["value"], groupby="group", nonnormal=["value"], pval=True)
        expected = stats.kruskal([1, 1, 2, 3], [2, 4, 8, 16], [3, 9, 27, 81]).pvalue

        assert table._htest_table.loc["value", "Test"] == "Kruskal-Wallis"
        assert table._htest_table.loc["value", "P-Value"] == pytest.approx(expected)

    def test_kruskal_for_two_nonnormal_groups(self):
        df = pd.DataFrame({"group": ["a"] * 4 + ["b"] * 4, "value": [1, 1, 2, 3, 2, 4, 8, 16]})

        table = TableOne(df, columns=["value"], groupby="group", nonnormal=["value"], pval=True)
        expected = stats.kruskal([1, 1, 2, 3], [2, 4, 8, 16]).pvalue

        assert table._htest_table.loc["value", "Test"] == "Kruskal-Wallis"
        assert table._htest_table.loc["value", "P-Value"] == pytest.approx(expected)

    def test_continuous_pvalue_not_tested_with_zero_observation_group(self):
        df = pd.DataFrame({"group": ["a", "a", "b", "b"], "value": [1, 2, None, None]})

        table = TableOne(df, columns=["value"], groupby="group", pval=True)

        assert table._htest_table.loc["value", "Test"] == "Not tested"
        assert pd.isnull(table._htest_table.loc["value", "P-Value"])

    def test_normal_continuous_pvalue_not_tested_with_single_observation_group(self):
        df = pd.DataFrame({"group": ["a", "b", "b", "b"], "value": [1, 2, 3, 4]})

        table = TableOne(df, columns=["value"], groupby="group", pval=True)

        assert table._htest_table.loc["value", "Test"] == "Not tested"
        assert pd.isnull(table._htest_table.loc["value", "P-Value"])

    def test_chi_square_for_categorical_groups_with_sufficient_counts(self):
        df = pd.DataFrame(
            {
                "group": ["a"] * 40 + ["b"] * 40,
                "color": ["red"] * 25 + ["blue"] * 15 + ["red"] * 12 + ["blue"] * 28,
            }
        )

        table = TableOne(df, columns=["color"], categorical=["color"], groupby="group", pval=True)
        contingency = [[15, 25], [28, 12]]
        expected = stats.chi2_contingency(contingency).pvalue

        assert table._htest_table.loc["color", "Test"] == "Chi-squared"
        assert table._htest_table.loc["color", "P-Value"] == pytest.approx(expected)

    def test_fisher_exact_pvalue_for_two_by_two_low_expected_counts(self):
        df = pd.DataFrame({"group": ["a", "a", "b", "b"], "color": ["red", "red", "blue", "blue"]})

        table = TableOne(df, columns=["color"], categorical=["color"], groupby="group", pval=True)
        contingency = [[0, 2], [2, 0]]
        expected = stats.fisher_exact(contingency).pvalue

        assert table._htest_table.loc["color", "Test"] == "Fisher's exact"
        assert table._htest_table.loc["color", "P-Value"] == pytest.approx(expected)

    def test_categorical_pvalue_includes_zero_cells_for_observed_groups(self):
        df = pd.DataFrame(
            {"group": ["a", "a", "a", "b", "b", "b"], "color": ["red", "red", "red", "red", "blue", "blue"]}
        )

        table = TableOne(df, columns=["color"], categorical=["color"], groupby="group", pval=True)
        contingency = [[0, 3], [2, 1]]
        expected = stats.fisher_exact(contingency).pvalue

        assert table._htest_table.loc["color", "Test"] == "Fisher's exact"
        assert table._htest_table.loc["color", "P-Value"] == pytest.approx(expected)

    def test_categorical_pvalue_not_tested_when_group_has_all_missing_values(self):
        df = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b", "c", "c"],
                "color": ["red", "blue", None, None, "red", "red"],
            }
        )

        table = TableOne(df, columns=["color"], categorical=["color"], groupby="group", pval=True)

        assert table._htest_table.loc["color", "Test"] == "Not tested"
        assert pd.isnull(table._htest_table.loc["color", "P-Value"])

    def test_categorical_pvalue_not_tested_when_two_group_one_group_all_missing(self):
        df = pd.DataFrame({"group": ["a", "a", "b", "b"], "color": ["red", "blue", None, None]})

        table = TableOne(df, columns=["color"], categorical=["color"], groupby="group", pval=True)

        assert table._htest_table.loc["color", "Test"] == "Not tested"
        assert pd.isnull(table._htest_table.loc["color", "P-Value"])

    def test_categorical_pvalue_ignores_unused_pandas_categories(self):
        color = pd.Categorical(["red", "red", "blue", "blue"], categories=["red", "blue", "green"])
        df = pd.DataFrame({"group": ["a", "a", "b", "b"], "color": color})

        table = TableOne(df, columns=["color"], categorical=["color"], groupby="group", pval=True)
        contingency = [[0, 2], [2, 0]]
        expected = stats.fisher_exact(contingency).pvalue

        assert table._htest_table.loc["color", "Test"] == "Fisher's exact"
        assert table._htest_table.loc["color", "P-Value"] == pytest.approx(expected)

    def test_pval_adjust_excludes_not_tested_variables(self):
        df = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b", "a", "b"],
                "ok": [1, 2, 3, 4, 5, 6],
                "bad": [1, 2, None, None, 3, None],
            }
        )

        table = TableOne(df, columns=["ok", "bad"], groupby="group", pval=True, pval_adjust="bonferroni")

        assert table._htest_table.loc["ok", "P-Value (adjusted)"] == pytest.approx(
            table._htest_table.loc["ok", "P-Value"]
        )
        assert pd.isnull(table._htest_table.loc["bad", "P-Value (adjusted)"])
        assert table.tableone["Grouped by group"].loc[("bad, mean (SD)", ""), "P-Value (adjusted)"] == ""

    def test_not_tested_raw_pvalue_displays_blank(self):
        df = pd.DataFrame({"group": ["a", "a", "b", "b"], "value": [1, 2, None, None]})

        table = TableOne(df, columns=["value"], groupby="group", pval=True)

        assert table.tableone["Grouped by group"].loc[("value, mean (SD)", ""), "P-Value"] == ""

    def test_custom_pvalues_participate_in_adjustment(self):
        def low(*groups):
            return 0.01

        def medium(*groups):
            return 0.02

        def high(*groups):
            return 0.2

        df = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b"],
                "low": [1, 2, 3, 4],
                "medium": [1, 2, 3, 4],
                "high": [1, 2, 3, 4],
            }
        )
        table = TableOne(
            df,
            columns=["low", "medium", "high"],
            groupby="group",
            pval=True,
            pval_adjust="bonferroni",
            htest={"low": low, "medium": medium, "high": high},
        )

        assert table._htest_table.loc["low", "P-Value (adjusted)"] == pytest.approx(0.03)
        assert table._htest_table.loc["medium", "P-Value (adjusted)"] == pytest.approx(0.06)
        assert table._htest_table.loc["high", "P-Value (adjusted)"] == pytest.approx(0.6)

    def test_custom_statistical_test_must_return_scalar_pvalue(self):
        def test_result(*groups):
            return stats.ttest_ind(*groups, equal_var=False)

        df = pd.DataFrame({"group": ["a", "a", "b", "b"], "value": [1, 2, 3, 4]})

        with pytest.raises(InputError):
            TableOne(df, columns=["value"], groupby="group", pval=True, htest={"value": test_result})

    def test_adjust_method_column_internal_only(self):
        df = pd.DataFrame({"group": ["a", "a", "b", "b"], "value": [1, 2, 3, 4]})

        table = TableOne(df, columns=["value"], groupby="group", pval=True, pval_adjust="bonferroni")

        assert "adjust method" in table._htest_table.columns
        assert "adjust method" not in table.tableone["Grouped by group"].columns

    def test_pval_adjusted_missing_row_does_not_include_raw_pvalue_column(self):
        df = pd.DataFrame(
            {
                "numbers": [1, 2, 6, 1, 1, 1],
                "colors": ["red", "white", "blue", "red", "blue", "blue"],
                "even": ["yes", "no", "yes", "yes", "no", "yes"],
            }
        )

        table = TableOne(df, groupby="even", pval=True, pval_adjust="bonferroni", missing_value_on_separate_row=True)

        assert "P-Value (adjusted)" in table.tableone["Grouped by even"].columns
        assert "P-Value" not in table.tableone["Grouped by even"].columns

    def test_small_pvalue_and_threshold_formatting(self):
        df = pd.DataFrame({"group": ["a"] * 5 + ["b"] * 5, "value": [1, 1, 2, 2, 3, 100, 101, 101, 102, 102]})

        table = TableOne(df, columns=["value"], groupby="group", pval=True, pval_threshold=0.05)

        assert table.tableone["Grouped by group"].loc[("value, mean (SD)", ""), "P-Value"] == "<0.001*"

    def test_small_adjusted_pvalue_and_threshold_formatting(self):
        df = pd.DataFrame({"group": ["a"] * 5 + ["b"] * 5, "value": [1, 1, 2, 2, 3, 100, 101, 101, 102, 102]})

        table = TableOne(
            df, columns=["value"], groupby="group", pval=True, pval_adjust="bonferroni", pval_threshold=0.05
        )

        assert table.tableone["Grouped by group"].loc[("value, mean (SD)", ""), "P-Value (adjusted)"] == "<0.001*"

    def test_pval_threshold_does_not_mark_not_tested_values(self):
        df = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b", "a", "b"],
                "ok": [1, 2, 3, 4, 5, 6],
                "bad": [1, 2, None, None, 3, None],
            }
        )

        table = TableOne(
            df, columns=["ok", "bad"], groupby="group", pval=True, pval_adjust="bonferroni", pval_threshold=0.05
        )

        assert table.tableone["Grouped by group"].loc[("bad, mean (SD)", ""), "P-Value (adjusted)"] == ""

    def test_custom_statistical_tests(self):
        func = mytest

        np.random.seed(12345678)
        n1 = 200
        n2 = 300

        rvs1 = stats.norm.rvs(size=n1, loc=0.0, scale=1)
        df1 = pd.DataFrame({"rvs": "rvs1", "val": rvs1})

        rvs2 = stats.norm.rvs(size=n2, loc=0.5, scale=1.5)
        df2 = pd.DataFrame({"rvs": "rvs2", "val": rvs2})

        rvs3 = stats.norm.rvs(size=n2, loc=0.01, scale=1.0)
        df3 = pd.DataFrame({"rvs": "rvs3", "val": rvs3})

        rvs4 = stats.norm.rvs(size=n2, loc=0.0, scale=1.0)
        df4 = pd.DataFrame({"rvs": "rvs4", "val": rvs4})

        different = pd.concat([df1, df2], ignore_index=True)
        t1_diff = TableOne(data=different, columns=["val"], pval=True, groupby="rvs", htest={"val": func})

        assert isclose(t1_diff._htest_table["P-Value"].val, stats.ks_2samp(rvs1, rvs2)[1])
        assert t1_diff._htest_table["Test"].val == "Test name"

        similar = pd.concat([df1, df3], ignore_index=True)
        t1_similar = TableOne(data=similar, columns=["val"], pval=True, groupby="rvs", htest={"val": func})

        assert isclose(t1_similar._htest_table["P-Value"].val, stats.ks_2samp(rvs1, rvs3)[1])

        identical = pd.concat([df1, df4], ignore_index=True)
        t1_identical = TableOne(data=identical, columns=["val"], pval=True, groupby="rvs", htest={"val": func})

        assert isclose(t1_identical._htest_table["P-Value"].val, stats.ks_2samp(rvs1, rvs4)[1])

    def test_custom_statistical_test_receives_groups_in_display_order(self):
        observed = []

        def custom_test(*groups):
            observed.extend([group.tolist() for group in groups])
            return 0.123

        df = pd.DataFrame({"group": ["b", "a", "b", "a"], "value": [10, 1, 20, 2]})
        table = TableOne(df, columns=["value"], groupby="group", pval=True, htest={"value": custom_test})

        assert observed == [[1, 2], [10, 20]]
        assert table._htest_table.loc["value", "P-Value"] == 0.123
        assert table._htest_table.loc["value", "Test"] == "custom_test"

    def test_custom_statistical_test_respects_numeric_group_order(self):
        observed = []

        def custom_test(*groups):
            observed.extend([group.tolist() for group in groups])
            return 0.123

        df = pd.DataFrame({"group": [0, 1, 0, 1], "value": [1, 10, 2, 20]})
        table = TableOne(
            df, columns=["value"], groupby="group", pval=True, htest={"value": custom_test}, order={"group": [1, 0]}
        )

        assert observed == [[10, 20], [1, 2]]
        assert list(table.tableone["Grouped by group"].columns) == ["Missing", "Overall", "1", "0", "P-Value"]

    def test_continuous_pvalue_rejects_non_numeric_values_before_testing(self):
        df = pd.DataFrame({"group": ["a", "a", "b", "b"], "value": [1, 2, 3, "bad"]})

        with pytest.raises(InputError):
            TableOne(df, columns=["value"], categorical=[], groupby="group", pval=True)
