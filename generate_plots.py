from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.formula.api as smf
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt


def solve_quadratic_inequality(a: float, b: float, c: float) -> List:
    """
    Solves a quadratic inequality of the form ax^2 + bx + c <= 0 or >= 0.

    Args:
        a (float): Coefficient of x^2.
        b (float): Coefficient of x.
        c (float): Constant term.

    Returns:
        List[Tuple[float, float]]: List of intervals where the inequality holds.
    """
    determinant = b**2 - 4 * a * c

    if determinant > 0:
        root2 = (-b + np.sqrt(determinant)) / (2 * a)
        root1 = (-b - np.sqrt(determinant)) / (2 * a)
        if a > 0:  # happy quadratic (parabola opens upwards)
            return [(root1, root2)]
        else:  # sad quadratic (parabola opens downwards)
            return [(-np.inf, root2), (root1, np.inf)]
    elif determinant < 0:
        if a > 0:  # parabola opens upwards, no real roots
            return []
        else:  # parabola opens downwards, always <= 0
            return [(-np.inf, np.inf)]
    elif determinant == 0:
        root = -b / (2 * a)
        if a > 0:  # parabola touches x-axis at one point
            return [(root, root)]
        else:  # parabola is always <= 0
            return [(-np.inf, np.inf)]


def generate_weakiv_data(
    n_samples: int, true_ATE: float, instrument_size: float, u_std: float
) -> Dict[str, np.ndarray]:
    """
    Generates weak instrument data for simulation.

    Args:
        n_samples (int): Number of samples.
        true_ATE (float): True average treatment effect.
        instrument_size (float): Size of the instrument effect.
        u_std (float): Standard deviation of the error term.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing generated data (Y, Z, W).
    """
    u = np.random.normal(0, u_std, size=n_samples)
    Z = np.random.binomial(1, 0.5, size=n_samples)
    W = instrument_size * Z + u
    W = np.array(W > 0, dtype=int)
    Y = true_ATE * W + u
    return {"Y": Y, "Z": Z, "W": W}


def bonferroni_confidence_interval(
    df: pd.DataFrame, alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Computes the Bonferroni confidence interval.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        alpha (float): Significance level.

    Returns:
        Tuple[float, float]: Confidence interval bounds.
    """
    confidence_bonferroni = alpha / 2
    mod_denominator = smf.ols("W ~ Z", data=df)
    res_denominator = mod_denominator.fit(cov_type="HC1")
    mod_numerator = smf.ols("Y ~ Z", data=df)
    res_numerator = mod_numerator.fit(cov_type="HC1")

    lower_bound_denominator = res_denominator.conf_int(confidence_bonferroni).iloc[1, 0]
    upper_bound_denominator = res_denominator.conf_int(confidence_bonferroni).iloc[1, 1]
    lower_bound_numerator = res_numerator.conf_int(confidence_bonferroni).iloc[1, 0]
    upper_bound_numerator = res_numerator.conf_int(confidence_bonferroni).iloc[1, 1]
    sign_denominator = lower_bound_denominator * upper_bound_denominator

    if sign_denominator > 0:
        if lower_bound_denominator > 0 and lower_bound_numerator > 0:
            upper_bound = upper_bound_numerator / lower_bound_denominator
            lower_bound = lower_bound_numerator / upper_bound_denominator
        if lower_bound_denominator < 0 and lower_bound_numerator < 0:
            upper_bound = lower_bound_numerator / upper_bound_denominator
            lower_bound = upper_bound_numerator / lower_bound_denominator
        if lower_bound_denominator < 0 and lower_bound_numerator > 0:
            upper_bound = lower_bound_numerator / lower_bound_denominator
            lower_bound = upper_bound_numerator / upper_bound_denominator
        if lower_bound_denominator > 0 and lower_bound_numerator < 0:
            upper_bound = upper_bound_numerator / upper_bound_denominator
            lower_bound = lower_bound_numerator / lower_bound_denominator
        conf_int = (lower_bound, upper_bound)
    else:
        conf_int = (-np.inf, np.inf)

    return conf_int


def get_psi_a(df: pd.DataFrame) -> np.ndarray:
    """
    Computes the psi_a term for Anderson-Rubin confidence intervals, where the score is written as
    psi_a phi + psi_b, and phi is the estimand of interest.

    Args:
        df (pd.DataFrame): Dataframe containing the data.

    Returns:
        np.ndarray: Computed psi_a values.
    """
    psi_a = -1 * np.multiply(
        2 * df["Z"] - 1, df["Y"] - np.mean(df["Y"] * df["Z"]) / np.mean(df["Z"])
    )
    psi_a = np.divide(psi_a, np.mean(df["Z"]))
    return psi_a


def get_psi_b(df: pd.DataFrame) -> np.ndarray:
    """
    Computes the psi_b term for Anderson-Rubin confidence intervals, where the score is written as
    psi_a phi + psi_b, and phi is the estimand of interest.

    Args:
        df (pd.DataFrame): Dataframe containing the data.

    Returns:
        np.ndarray: Computed psi_b values.
    """
    psi_b = np.multiply(
        2 * df["Z"] - 1, df["W"] - np.mean(df["W"] * df["Z"]) / np.mean(df["Z"])
    )
    psi_b = np.divide(psi_b, np.mean(df["Z"]))
    return psi_b


def anderson_rubin_confidence_interval(
    df: pd.DataFrame, alpha: float = 0.05
) -> List[Tuple[float, float]]:
    """
    Computes the Anderson-Rubin confidence interval.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        alpha (float): Significance level.

    Returns:
        List[Tuple[float, float]]: List of confidence intervals.
    """
    n = len(df)
    critical_value = norm.ppf(1 - alpha / 2)
    psi_a = get_psi_a(df)
    psi_b = get_psi_b(df)
    a = n * np.mean(psi_a) ** 2 - critical_value**2 * np.mean(np.square(psi_a))
    b = 2 * n * np.mean(psi_a) * np.mean(psi_b) - 2 * critical_value**2 * np.mean(
        np.multiply(psi_a, psi_b)
    )
    c = n * np.mean(psi_b) ** 2 - critical_value**2 * np.mean(np.square(psi_b))
    conf_set = solve_quadratic_inequality(a, b, c)
    return conf_set


def run_simulation(
    n_samples: int, true_ATE: float, instrument_size: float, u_std: float
) -> Tuple[bool, bool, float, float]:
    """
    Runs a single simulation to compute coverage and interval lengths.

    Args:
        n_samples (int): Number of samples.
        true_ATE (float): True average treatment effect.
        instrument_size (float): Size of the instrument effect.
        u_std (float): Standard deviation of the error term.

    Returns:
        Tuple[bool, bool, float, float]: Coverage and interval lengths for score and Bonferroni methods.
    """
    # Generate data
    data_dict = generate_weakiv_data(n_samples, true_ATE, instrument_size, u_std)
    df = pd.DataFrame(data_dict)

    score_conf_set = anderson_rubin_confidence_interval(df)
    bonferroni_conf_interval = bonferroni_confidence_interval(df)

    coverage_score = (
        any(lower[0] <= true_ATE <= lower[1] for lower in score_conf_set)
        if score_conf_set
        else False
    )
    coverage_bonferroni = (
        (bonferroni_conf_interval[0] <= true_ATE <= bonferroni_conf_interval[1])
        if bonferroni_conf_interval
        else False
    )
    length_score = (
        max([inter[1] - inter[0] for inter in score_conf_set]) if score_conf_set else 0
    )
    length_bonferroni = bonferroni_conf_interval[1] - bonferroni_conf_interval[0]

    return coverage_score, coverage_bonferroni, length_score, length_bonferroni


if __name__ == "__main__":

    np.random.seed(42)
    n_samples = [1500, 2000, 3000, 5000, 10000]
    true_ATE = 1

    n_simulations = 2000
    u_std = 1

    output = []

    for n in n_samples:
        instrument_size = 50 / np.sqrt(n)
        print(
            f"Running simulation for n_samples: {n}, instrument_size: {instrument_size}"
        )
        for _ in tqdm(range(n_simulations)):
            coverage_score, coverage_bonferroni, length_score, length_bonferroni = (
                run_simulation(n, true_ATE, instrument_size, u_std)
            )
            output.append(
                {
                    "n_samples": n,
                    "instrument_size": instrument_size,
                    "true_ATE": true_ATE,
                    "coverage_score": coverage_score,
                    "coverage_bonferroni": coverage_bonferroni,
                    "length_score": length_score,
                    "length_bonferroni": length_bonferroni,
                }
            )

    output_df = pd.DataFrame(output)

    output_means = output_df.groupby("n_samples").mean()

    # Create a seaborn plot
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=output_means,
        x=output_means.index,
        y="coverage_score",
        marker="o",
        linestyle="dotted",
        label="Score confidence set",
        color="black",
    )
    sns.lineplot(
        data=output_means,
        x=output_means.index,
        y="coverage_bonferroni",
        marker="X",
        linestyle="-.",
        label="Bonferroni confidence set",
        color="black",
    )

    # Add a dashed line at 0.95
    plt.axhline(0.95, linestyle="--", label="Nominal Coverage (0.95)", color="black")

    # Customize the plot
    plt.xlabel("Sample size")
    plt.ylabel("Coverage")
    plt.title("Coverage vs. sample size")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.savefig("coverage_vs_sample_size.png")

    output_medians = output_df.groupby("n_samples").median()

    # Create a seaborn plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=output_means,
        x=output_means.index,
        y="length_score",
        marker="o",
        linestyle="dotted",
        label="Score confidence set",
        color="black",
    )
    sns.lineplot(
        data=output_means,
        x=output_means.index,
        y="length_bonferroni",
        marker="X",
        linestyle="-.",
        label="Bonferroni confidence set",
        color="black",
    )

    # Customize the plot
    plt.xlabel("Sample size")
    plt.ylabel("Median length")
    plt.title("Median length vs. sample size")
    plt.legend()
    plt.grid(True)

    # Show the plot

    plt.savefig("median_length_vs_sample_size.png")
