import polars as pl
from scipy.stats import chi2_contingency

transactions = pl.scan_parquet("data/transactions_train_3.parquet")

transactions_labeled = transactions.filter(pl.col("label") != "UNKNOWN").select(
    (
        pl.col("customer_feedback").is_not_null().cast(pl.Int8).alias("has_feedback"),
        pl.col("label"),
    )
)


def chi_test(df: pl.DataFrame, col1: str, col2: str) -> float:
    """
    Perform a Chi-squared test of independence on two categorical columns in a DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame.
    col1 : str
        The first categorical column.
    col2 : str
        The second categorical column.

    Returns
    -------
    float
        The p-value from the Chi-squared test.
    """
    contingency_table = (
        df.group_by([col1, col2])
        .agg(pl.len().alias("count"))
        .pivot(values="count", index=col1, on=col2)
    )
    print(f"Contingency table:\n{contingency_table}")
    contingency_table.to_numpy()
    chi2, p, dof, exp = chi2_contingency(contingency_table)
    print(f"Chi-squared test result: chi2={chi2}, p-value={p}, dof={dof}")
    # print expected frequencies in the same forat as the contingency table
    exp = pl.DataFrame(exp, schema=contingency_table.columns)
    print(f"Expected frequencies:\n{exp}")
    return p


chi_test(
    transactions_labeled.collect(),
    "has_feedback",
    "label",
)
