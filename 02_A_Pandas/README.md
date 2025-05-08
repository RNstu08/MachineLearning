# Pandas Learning Roadmap

This roadmap outlines the key concepts and functionalities within the Pandas library, designed to take you from foundational knowledge to advanced usage.

## I. Introduction & Core Data Structures

* **What is Pandas?** Purpose, advantages (high-performance data structures, analysis tools), relationship with NumPy.
* **Pandas Series:** 1D labeled array (like a column).
    * Creation: `pd.Series()` (from lists, dicts, NumPy arrays, scalars).
    * Attributes: `.index`, `.values`, `.dtype`, `.name`, `.size`, `.shape`.
    * Index Object: Understanding the index.
    * Basic Operations: Vectorized arithmetic, boolean indexing, accessing elements.
* **Pandas DataFrame:** 2D labeled data structure (like a table).
    * Creation: `pd.DataFrame()` (from dicts of lists/Series, list of dicts, NumPy arrays, Series, another DataFrame).
    * Attributes: `.index`, `.columns`, `.values`, `.dtypes`, `.shape`, `.size`.
    * Basic Inspection: `.head()`, `.tail()`, `.info()`, `.describe()`.

## II. Data Loading & Saving (I/O)

* **Reading Data:** Importing data into DataFrames.
    * **CSV Files:** `pd.read_csv()` (key parameters: `filepath_or_buffer`, `sep`, `header`, `index_col`, `usecols`, `dtype`, `parse_dates`, `nrows`, `skiprows`).
    * **Excel Files:** `pd.read_excel()` (key parameters: `io`, `sheet_name`, `header`, `index_col`, `usecols`, `dtype`, `parse_dates`).
    * **Other Formats:** `pd.read_json()`, `pd.read_sql()`, `pd.read_html()`, `pd.read_clipboard()`, `pd.read_parquet()`, `pd.read_feather()`, `pd.read_hdf()`, etc.
* **Writing Data:** Exporting DataFrames.
    * `df.to_csv()`, `df.to_excel()`, `df.to_json()`, `df.to_sql()`, `df.to_parquet()`, etc. (key parameters: `path_or_buf`, `index`, `header`, `sep`, `columns`, `mode`, `encoding`).

## III. Data Inspection & Selection

* **Viewing/Inspecting Data:** `.head()`, `.tail()`, `.sample()`, `.info()`, `.describe()`, `.shape`, `.dtypes`, `.columns`, `.index`, `.values`.
* **Summarizing Data:** `.nunique()`, `.value_counts()`.
* **Selection Techniques:**
    * **Column Selection:** Bracket notation (`df['col']`, `df[['col1', 'col2']]`), Dot notation (`df.col` - use cautiously).
    * **Row Selection (Label-based):** `.loc[]` (single label, list of labels, slice of labels, boolean array).
    * **Row Selection (Position-based):** `.iloc[]` (single integer, list of integers, slice of integers, boolean array - positional).
    * **Combined Selection:** Using `.loc`/`.iloc` for specific row/column intersections.
    * **Conditional Selection (Boolean Indexing):** `df[boolean_condition]`, using `&` (and), `|` (or), `~` (not), `.isin()`, `.between()`.
* **Setting Values:** Assigning new values using selection methods (`.loc`, `.iloc`, boolean indexing).
* **Index Manipulation:** `df.set_index()`, `df.reset_index()`.

## IV. Data Cleaning & Preparation

* **Handling Missing Data (NaN):**
    * Identifying: `.isnull()`, `.isna()`, `.notnull()`, `.notna()`, `.isnull().sum()`.
    * Dropping: `.dropna()` (parameters: `axis`, `how`, `thresh`, `subset`).
    * Filling: `.fillna()` (parameters: `value`, `method='ffill'/'bfill'`, using aggregates like mean/median/mode, per-column filling).
* **Handling Duplicates:** `.duplicated()`, `.drop_duplicates()` (parameters: `subset`, `keep`).
* **Data Type Conversion:** `.astype()`.
* **Renaming Columns & Index:** `.rename()`.
* **Applying Functions:**
    * Element-wise: `.map()` (for Series), `.applymap()` (for DataFrame - use less often).
    * Row/Column-wise: `.apply()` (with `axis=0` or `axis=1`, custom functions, lambda functions).
* **Replacing Values:** `.replace()`.
* **Sorting:** `.sort_values()` (by column(s)), `.sort_index()`.

## V. Data Wrangling: Combining & Reshaping

* **Combining DataFrames:**
    * **Concatenation (stacking):** `pd.concat()` (parameters: `axis=0` or `axis=1`, `join='inner'/'outer'`, `ignore_index`).
    * **Merging/Joining (database-style):** `pd.merge()` (parameters: `on`, `left_on`, `right_on`, `how='inner'/'outer'/'left'/'right'`, `suffixes`), `df.join()` (joins based on index by default).
* **Reshaping Data:**
    * **Pivoting (long to wide):** `.pivot()` (`index`, `columns`, `values`), `pd.pivot_table()` (handles duplicates via aggregation - `aggfunc`).
    * **Melting (wide to long):** `pd.melt()` (`id_vars`, `value_vars`, `var_name`, `value_name`).
    * **Stacking/Unstacking (hierarchical index):** `.stack()`, `.unstack()`.

## VI. Grouping & Aggregation (Groupby Operations)

* **The Split-Apply-Combine Strategy:** Understanding the `.groupby()` process.
* **Grouping:** `df.groupby()` (by single column, multiple columns, Series, mapping, level).
* **Aggregation (`.agg()` or `.aggregate()`):** Applying functions to groups.
    * Built-in methods: `.sum()`, `.mean()`, `.median()`, `.count()`, `.size()`, `.std()`, `.var()`, `.min()`, `.max()`, `.first()`, `.last()`, `.nunique()`.
    * Using `.agg()`: With single function, list of functions, dict of `{column: function(s)}`.
* **Transformation (`.transform()`):** Applying a function group-wise but returning results aligned with the original DataFrame's index.
* **Filtering (`.filter()`):** Keeping/discarding entire groups based on a group-level condition.

## VII. Time Series Analysis

* **Date/Time Data Types:** `datetime64[ns]`, `Timestamp`, `Timedelta`, `Period`.
* **Creating Date Ranges & Conversion:** `pd.to_datetime()`, `pd.date_range()`.
* **Time Series Indexing:** Using `DatetimeIndex`, partial string indexing, slicing.
* **Time Zone Handling:** `.tz_localize()`, `.tz_convert()`.
* **Resampling:** `.resample()` (changing frequency, downsampling/upsampling) with aggregation (`.sum()`, `.mean()`, `.ohlc()`, etc.).
* **Shifting/Lagging:** `.shift()`.
* **Rolling Windows:** `.rolling()` (for moving window calculations: `mean`, `sum`, `std`, etc.).
* **Expanding Windows:** `.expanding()` (cumulative calculations).

## VIII. Working with Text & Categorical Data

* **String Methods (`.str` accessor):** Accessing vectorized string functions (`.lower()`, `.upper()`, `.strip()`, `.split()`, `.join()`, `.contains()`, `.startswith()`, `.endswith()`, `.replace()`, `.findall()`, `.extract()` (regex), `.len()`, etc.).
* **Categorical Data:** Using the `Categorical` dtype for memory efficiency and performance (`.astype('category')`, `.cat` accessor).

## IX. Visualization with Pandas

* **Basic Plotting:** `df.plot()` method and `Series.plot()` (`kind='line'`, `'bar'`, `'barh'`, `'hist'`, `'box'`, `'kde'/'density'`, `'area'`, `'pie'`, `'scatter'`, `'hexbin'`). Relies on Matplotlib.
* **Customization:** Passing arguments to underlying Matplotlib functions.
* **Integration:** Using Pandas with Matplotlib and Seaborn for more advanced visualizations.

## X. Advanced Topics & Performance

* **MultiIndex (Hierarchical Indexing):** Creating, selecting, slicing, and working with multiple index levels.
* **Performance Optimization:**
    * Using efficient data types (`category`, smaller numeric types).
    * Vectorization (avoiding Python loops, `.apply()` when possible).
    * Using `.eval()` and `.query()` for faster expression evaluation.
    * Reading/Writing efficient file formats (`Parquet`, `Feather`, `HDF5`).
    * Reading large files in chunks (`chunksize` parameter).
* **Options & Settings:** Customizing Pandas behavior with `pd.set_option()`.
* **Extending Pandas:** Brief overview of custom accessors/methods.