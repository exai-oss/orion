"""Scripts for model performance reporting.

Copyright (C) 2024 Exai Bio Inc. Authors

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from statsmodels.stats.proportion import proportion_confint


def plot_stratified_sensitivities(
    df_in: pd.DataFrame,
    sample_map: dict[str, str],
    ax: plt.Axes,
    xlabel: str,
    fpr: float = 0.05,
    labels: Optional[dict[str, str]] = None,
    fontsize: int = 10,
) -> dict[str, tuple]:
    """Plots stratifies sensitivty values.

    Args:
        df_in: A pd.DataFrame with 'sample_id' index, 'score' (float) and
            'label' (binary) columns.
        sample_map: A mapping of each 'sample_id' to a category. One of the
            categories must be 'normal'.
        ax: Axes for plotting the sensitivities.
        xlabel: The X-axis label for the plot.
        fpr: The desired False Positive Rate.
        labels: A mapping of categories to their plottinglabels, if None,
            `sample_map.values()` will be used.
        fontsize: Font size to use.

    Raises:
        ValueError: If `sample_map` does not have 'normal' in its `.values()`.

    Returns:
        A dict mapping categories to their sensitivity point estimates and
        confidence intervals.

    """

    categories = sorted(set(sample_map.values()))
    if "normal" not in categories:
        raise ValueError("No 'normal' samples are provided.")

    categories.remove("normal")
    num_cats = len(categories)
    categories_xcoords = list(1 + np.arange(num_cats))

    ticks_fontsize = fontsize - 2

    sample_map_df = pd.DataFrame.from_dict(
        sample_map, orient="index", columns=["category"]
    )
    df = pd.merge(df_in, sample_map_df, left_index=True, right_index=True)

    # compute sensitivities
    category_info = {}
    for category in categories:
        df_category = df.query(
            f"category == '{category}' or category == 'normal'"
        ).copy()
        category_info[category] = find_sensitivity_linear(df_category, fpr)
    ax.set_ylim(ymin=0, ymax=1.1)
    ax.set_xlim(xmin=0.5, xmax=num_cats + 0.5)

    xtick_labels = []
    for category, categories_xcoord in zip(categories, categories_xcoords):
        plot_confidence_interval_clopper(
            categories_xcoord,
            category_info[category],
            ax,
            fontsize=ticks_fontsize,
        )

        if labels is None:
            xtick_labels.append(f"{category} ({category_info[category][2]})")
        else:
            xtick_labels.append(
                f"{labels[category]} ({category_info[category][2]})"
            )

    ax.set_xticks(categories_xcoords)
    ax.set_yticks(np.arange(0, 1.2, step=0.25))
    ax.set_xticklabels(xtick_labels, fontsize=ticks_fontsize)

    ax.tick_params(axis="both", labelsize=ticks_fontsize)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(
        f"Sensitivity at {100*(1 - fpr):.0f}% Specificity", fontsize=fontsize
    )
    ax.grid(visible=True)
    return category_info


def plot_confidence_interval_clopper(
    x: float,
    sens_info: tuple[float, tuple[float, float], int],
    ax: plt.Axes,
    color: str = "navy",
    horizontal_line_width: float = 0.1,
    fontsize: int = 8,
) -> Union[float, tuple[float]]:
    """Plots the sensitivity and its confidence intervals for a category.

    Args:
        x: X-axis coordinate for plotting the bar.
        sens_info: A [float, (float, float)], containing the sensitivity
                   point estimate and lower higher-range of
                   the confidence interval.
        ax: Axes for plotting the bar.
        color: Color of the bar.
        horizontal_line_width: Width of the bar caps.
        fontsize: Font size for the text.

    Returns:
        Sensitivity point estimate.
    """
    # raises TypeError if incoming list is malformed
    point_estimate = sens_info[0]
    bottom, top = sens_info[1]

    # these are left and right coordinates of the caps
    left = x - horizontal_line_width / 2
    right = x + horizontal_line_width / 2

    ax.plot([x, x], [bottom, top], color=color)
    ax.plot([left, right], [top, top], color=color)
    ax.plot([left, right], [bottom, bottom], color=color)
    ax.plot(x, point_estimate, "o", color=color)
    ax.text(
        x + 0.075,
        point_estimate - 0.015,
        f"{point_estimate:.2f}",
        fontsize=fontsize,
    )
    return point_estimate


def find_sensitivity_linear(
    df_in: pd.DataFrame, fpr: float
) -> tuple[float, tuple[float, float], int]:
    """Find the interpolated TPR at a given FPR.

    Args:
        df_in: DataFrame with at least two columns: 'score' (continuous) and
            'label' (binary).
        fpr: Desired False Positive Rate.

    Raises:
        ValueError: If `df_in['label']` is all ones or zeros.

    Returns:
        A tuple consisting of, respectively: Sensitivity at `fpr`, an interior
        tuple of the Clopper-Pearson confidence interval, and the number of
        positives.
    """

    fprs, tprs, _ = sklearn.metrics.roc_curve(df_in["label"], df_in["score"])

    # find the index where <fpr> falls between two consecutive <fprs> elements.
    for i in range(len(fprs) - 1):
        less_than_fpr = fprs[i] <= fpr
        more_than_fpr = fprs[i + 1] > fpr
        if less_than_fpr and more_than_fpr:
            break

    fpr_low, tpr_low = fprs[i], tprs[i]
    fpr_high, tpr_high = fprs[i + 1], tprs[i + 1]

    less_than_fpr = fpr_low <= fpr
    more_than_fpr = fpr_high > fpr
    assert (
        less_than_fpr and more_than_fpr
    ), f"fpr range ({fpr_low}, {fpr_high}) is invalid."

    line = np.polyfit([fpr_low, fpr_high], [tpr_low, tpr_high], 1)

    # this gives the interpolated TPR at 1 - FPR specificity
    linear_tpr = line[0] * fpr + line[1]

    num_cancer = np.sum(df_in["label"] == 1)
    tp_round = np.round(num_cancer * linear_tpr)

    ci = proportion_confint(tp_round, num_cancer, alpha=0.05, method="beta")
    return linear_tpr, ci, num_cancer


def plot_roc(
    df_in: pd.DataFrame,
    ax: plt.Axes,
    title: Optional[str] = None,
    fpr: float = 0.05,
    fontsize: int = 10,
    linewidth: int = 2,
):
    """Plots a ROC curve.

    Args:
        df_in: DataFrame with 'score' (float) and 'label' (binary) columns.
        ax: Axes for plotting the ROC.
        title: Title of plot.
        fpr: False positive rate.
        fontsize: Font size.
        linewidth: Line width.
    """

    ticks_fontsize = fontsize - 2

    fprs, tprs, _ = sklearn.metrics.roc_curve(df_in["label"], df_in["score"])
    roc_auc = sklearn.metrics.auc(fprs, tprs)

    overall_sen = find_sensitivity_linear(df_in, fpr)

    ax.plot([0, 1], [0, 1], color="navy", lw=linewidth, linestyle="--")
    ax.plot(
        fprs,
        tprs,
        color="darkorange",
        lw=linewidth,
        label=f"ROC curve (area={roc_auc:.2f})",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.tick_params(axis="both", which="major", labelsize=ticks_fontsize)
    ax.plot(
        [fpr, fpr],
        [0, 1.5],
        color="red",
        lw=linewidth,
        linestyle="--",
        label=f"Specificity={100*(1 - fpr):.0f}%",
    )

    ax.text(
        fpr + 0.05,
        overall_sen[0] - 0.01,
        f"Sensitivity={100*(overall_sen[0]):.1f}%",
        fontsize=ticks_fontsize,
    )
    ax.set_xlabel("False Positive Rate", fontsize=fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=fontsize)
    ax.legend(loc="lower right", fontsize=ticks_fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
