# Seaborn Learning Roadmap

This roadmap outlines the key concepts and functionalities within the Seaborn library for creating informative and attractive statistical graphics in Python.

## I. Introduction & Setup

* **What is Seaborn?** Purpose (statistical data visualization), relationship with Matplotlib, key advantages (aesthetics, Pandas integration, statistical focus).
* **Installation:** `pip install seaborn`.
* **Importing Convention:** `import seaborn as sns`, `import matplotlib.pyplot as plt`.
* **Key Concepts:**
    * Figure-level vs. Axes-level functions.
    * Integration with Pandas DataFrames (long-form vs. wide-form data).
    * Semantic mapping (using variable names for plot aesthetics like `x`, `y`, `hue`, `style`, `size`).
* **Loading Example Datasets:** Using `sns.load_dataset()`.

## II. Controlling Plot Aesthetics

* **Seaborn Styles:** Setting the overall look (`sns.set_theme()` or `sns.set_style()`, e.g., `'darkgrid'`, `'whitegrid'`, `'ticks'`). Temporarily setting style with `with sns.axes_style():`.
* **Color Palettes:** Understanding and using different color palettes (`palette=` argument, `sns.color_palette()`, qualitative, sequential, diverging palettes). Tools like `sns.palplot()`.
* **Scaling Plot Elements:** Adjusting context for different output formats (`sns.set_context()`, e.g., `'paper'`, `'notebook'`, `'talk'`, `'poster'`).

## III. Visualizing Distributions (Univariate & Bivariate)

* **Axes-level Functions:**
    * Histograms: `sns.histplot()` (distribution shape, `bins`, `kde`, `stat`).
    * Kernel Density Estimates: `sns.kdeplot()` (smoothed distribution, `fill`, `bw_adjust`).
    * Empirical Cumulative Distribution Functions: `sns.ecdfplot()`.
    * Rug Plots: `sns.rugplot()` (individual observations).
* **Figure-level Function:** `sns.displot()` (combines the above with `FacetGrid` capabilities).
* **Bivariate Distributions:** `sns.kdeplot(x=..., y=...)`, `sns.histplot(x=..., y=...)`.

## IV. Visualizing Relationships

* **Axes-level Functions:**
    * Scatter Plots: `sns.scatterplot()` (relationship between two numerical variables, semantic mapping with `hue`, `size`, `style`).
    * Line Plots: `sns.lineplot()` (trends, often with time; handles aggregation and confidence intervals).
* **Figure-level Function:** `sns.relplot()` (combines `scatterplot` and `lineplot` with `FacetGrid`).
* **Regression Plots (Axes-level):** `sns.regplot()` (scatter plot with linear regression model fit and confidence interval).
* **Regression Plots (Figure-level):** `sns.lmplot()` (combines `regplot` with `FacetGrid`, allows conditioning on other variables).

## V. Visualizing Categorical Data

* **Axes-level Functions:**
    * Categorical Scatter Plots:
        * `sns.stripplot()` (points spread randomly).
        * `sns.swarmplot()` (points adjusted to avoid overlap - use for smaller datasets).
    * Categorical Distribution Plots:
        * `sns.boxplot()` (box-and-whisker plot).
        * `sns.violinplot()` (combines boxplot with KDE).
        * `sns.boxenplot()` (enhanced boxplot for larger datasets).
    * Categorical Estimate Plots:
        * `sns.pointplot()` (shows point estimates and confidence intervals).
        * `sns.barplot()` (shows point estimates - mean by default - and error bars).
        * `sns.countplot()` (shows counts of observations in each category).
* **Figure-level Function:** `sns.catplot()` (unified interface for all categorical plots combined with `FacetGrid`).

## VI. Visualizing Matrix Data

* **Heatmaps:** `sns.heatmap()` (visualizing matrix data, correlation matrices), annotations, colormaps, masking.
* **Clustermaps:** `sns.clustermap()` (hierarchically-clustered heatmap).

## VII. Multi-plot Grids for Exploring Relationships

* **`sns.pairplot()`:** Plots pairwise relationships across an entire DataFrame (scatter plots for variable pairs, distribution plots on the diagonal). Mapping `hue`.
* **`sns.jointplot()`:** Plots a bivariate relationship (`scatter`, `kde`, `hist`, `reg`) along with marginal univariate distributions.
* **`sns.FacetGrid`:** The underlying object used by figure-level functions. Can be used directly for custom multi-plot grids conditioned on data variables. Mapping plots onto the grid.

## VIII. Customization & Integration

* **Working with Matplotlib:** Accessing the underlying Matplotlib `Axes` from Seaborn plots (especially axes-level plots or grids) for further customization using Matplotlib functions (`ax.set_title`, `plt.xlabel`, etc.).
* **Customizing Legends, Titles, Labels:** Using Seaborn parameters and Matplotlib functions.
* **Saving Plots:** Using `plt.savefig()` or `figure.savefig()`.