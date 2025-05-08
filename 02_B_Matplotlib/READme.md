# Matplotlib Learning Roadmap

This roadmap outlines the essential concepts and functionalities within the Matplotlib library for creating static, animated, and interactive visualizations in Python.

## I. Introduction & Setup

* **What is Matplotlib?** Purpose (data visualization), history, role in the scientific Python ecosystem.
* **Architecture:** Understanding the backend, artist, and scripting layers (briefly).
* **Interfaces:**
    * `pyplot` interface (MATLAB-style state machine).
    * Object-Oriented (OO) interface (more explicit control, preferred for complex plots).
* **Installation:** `pip install matplotlib`.
* **Importing Convention:** `import matplotlib.pyplot as plt`.
* **Displaying Plots:** `plt.show()`, inline display in notebooks (`%matplotlib inline`).

## II. Basic Plotting with Pyplot

* **Simple Plots:** Creating basic line plots (`plt.plot()`) and scatter plots (`plt.scatter()`) from lists or NumPy arrays.
* **Plotting from Pandas:** Directly plotting Pandas Series and DataFrames.
* **Basic Customization:**
    * Labels and Title: `plt.xlabel()`, `plt.ylabel()`, `plt.title()`.
    * Legends: `plt.legend()`.
    * Colors, Linestyles, Markers: Arguments within plotting functions (e.g., `color=`, `linestyle=`, `marker=`).
    * Axis Limits: `plt.xlim()`, `plt.ylim()`.
    * Grid Lines: `plt.grid()`.

## III. The Object-Oriented Interface: Figure & Axes

* **Core Objects:** Understanding the `Figure` (the overall window/canvas) and `Axes` (individual plots/subplots).
* **Creating Figures & Axes:**
    * `plt.figure()`: Creates a new `Figure`.
    * `fig.add_subplot()` / `fig.add_axes()`: Adding `Axes` to a `Figure`.
    * `plt.subplots()`: Convenient function to create a `Figure` and a grid of `Axes` simultaneously (most common).
* **Plotting with Axes Methods:** Using methods directly on `Axes` objects (e.g., `ax.plot()`, `ax.scatter()`, `ax.hist()`).
* **Customizing with Axes Methods:** Setting labels (`ax.set_xlabel()`, `ax.set_ylabel()`), title (`ax.set_title()`), legends (`ax.legend()`), limits (`ax.set_xlim()`, `ax.set_ylim()`), ticks (`ax.set_xticks()`, `ax.set_yticks()`).
* **Why Use OO?** Better control, clarity for complex figures with multiple subplots.

## IV. Common Plot Types

* **Line Plots:** `ax.plot()` (time series, trends).
* **Scatter Plots:** `ax.scatter()` (relationships, distributions), customizing size (`s`) and color (`c`).
* **Bar Charts:** `ax.bar()` (vertical), `ax.barh()` (horizontal), comparing categorical data. Grouped and stacked bars.
* **Histograms:** `ax.hist()` (distributions of single variables), `bins`, density, cumulative.
* **Box Plots (Box-and-Whisker):** `ax.boxplot()` (distribution summaries, comparing groups).
* **Pie Charts:** `ax.pie()` (proportions - *use with caution*), labels, percentages, explode.
* **Error Bars:** `ax.errorbar()` (showing uncertainty).
* **Stem Plots:** `ax.stem()` (discrete sequences).
* **Fill Between:** `ax.fill_between()` / `ax.fill_betweenx()` (highlighting ranges).

## V. Customization & Styling

* **Colors:** Named colors, hex codes, RGB(A) tuples, colormaps (`cmap`).
* **Line Styles & Markers:** Detailed options.
* **Text & Annotations:** `ax.text()` (adding text at coordinates), `ax.annotate()` (adding text with arrows). LaTeX support for mathematical notation.
* **Ticks and Tick Labels:** `ax.set_xticks()`, `ax.set_yticks()`, `ax.set_xticklabels()`, `ax.set_yticklabels()`, `ticker` module (Locators and Formatters for fine control).
* **Axis Scales:** Linear (default), logarithmic (`ax.set_xscale('log')`, `ax.set_yscale('log')`).
* **Spines:** Controlling the visibility and position of plot boundaries.
* **Legends:** Customizing location, title, appearance.
* **Stylesheets:** Using pre-defined styles (`plt.style.use()`).
* **Configuration (`rcParams`):** Customizing default settings.

## VI. Working with Multiple Plots (Subplots)

* **`plt.subplots()` revisited:** Creating grids (`nrows`, `ncols`), sharing axes (`sharex`, `sharey`).
* **Accessing Axes Objects:** Indexing the array returned by `plt.subplots()`.
* **More Complex Layouts:** `GridSpec`, `fig.add_subplot()`.
* **Adjusting Layout:** `plt.tight_layout()`, `fig.tight_layout()`, `plt.subplots_adjust()`.

## VII. Advanced Plot Types

* **Images:** `ax.imshow()` (displaying 2D arrays like images, heatmaps), color mapping, interpolation. Colorbars (`plt.colorbar`/`fig.colorbar`).
* **Contour Plots:** `ax.contour()` (lines), `ax.contourf()` (filled), visualizing 3D data in 2D.
* **3D Plotting:** Using the `mpl_toolkits.mplot3d` toolkit (`Axes3D`) for scatter, surface, wireframe, contour plots in 3D.
* **Streamplots:** `ax.streamplot()` (visualizing vector fields).
* **Polar Plots:** Using `subplot_kw={'projection': 'polar'}`.

## VIII. Saving Plots

* **`plt.savefig()` / `fig.savefig()`:** Saving figures to files.
* **Parameters:** File path, supported formats (`PNG`, `JPG`, `PDF`, `SVG`, etc.), resolution (`dpi`), background color, bounding box (`bbox_inches='tight'`), transparency.

## IX. Integration & Ecosystem

* **Using Matplotlib with NumPy & Pandas:** Passing data structures directly.
* **Seaborn:** How Seaborn builds on Matplotlib for statistical visualization (brief comparison/mention).
* **Interactive Plotting:** Mention of backends like `ipympl` for interactive elements in Jupyter environments.