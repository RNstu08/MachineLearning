# NumPy Learning Roadmap

This roadmap outlines the essential concepts and functionalities within the NumPy library, designed to provide a complete hands-on understanding.

## I. Introduction & Fundamentals

* **What is NumPy?** Purpose, advantages (efficiency, speed), role in the scientific Python ecosystem.
* **The `ndarray` Object:** Core data structure (N-dimensional array), homogeneity (all elements same type).
* **Installation:** Brief mention (`pip install numpy`).
* **Importing Convention:** `import numpy as np`.
* **NumPy vs. Python Lists:** Key differences (performance, memory, functionality).

## II. Array Creation

* **From Python Structures:** `np.array()` (from lists, tuples).
* **Intrinsic Creation Functions:**
    * `np.arange()`: Evenly spaced values within a given interval (like Python's `range` but returns an array).
    * `np.linspace()`: Evenly spaced numbers over a specified interval (specifying the number of points).
    * `np.zeros()`, `np.zeros_like()`: Arrays filled with zeros.
    * `np.ones()`, `np.ones_like()`: Arrays filled with ones.
    * `np.full()`, `np.full_like()`: Arrays filled with an arbitrary constant.
    * `np.eye()`, `np.identity()`: Identity matrices.
    * `np.empty()`, `np.empty_like()`: "Empty" arrays (initial content is arbitrary).
* **Random Number Generation (`np.random`)**:
    * `np.random.rand()`: Uniform distribution [0, 1).
    * `np.random.randn()`: Standard normal distribution (mean 0, variance 1).
    * `np.random.randint()`: Random integers.
    * `np.random.choice()`: Random sampling from a given 1D array.
    * `np.random.seed()`: For reproducible results.
    * Other distributions (e.g., binomial, poisson, normal).

## III. Array Attributes & Data Types

* **Key Attributes:**
    * `ndarray.ndim`: Number of dimensions (axes).
    * `ndarray.shape`: Tuple representing array dimensions.
    * `ndarray.size`: Total number of elements.
    * `ndarray.dtype`: Data type of the array elements.
    * `ndarray.itemsize`: Size (in bytes) of each element.
    * `ndarray.nbytes`: Total bytes consumed by the array's elements.
* **Data Types (`dtype`)**: Understanding common types (`int32`, `int64`, `float32`, `float64`, `bool`, `complex`, `object`, `string_`, `unicode_`).
* **Type Casting:** `ndarray.astype()` method to convert between data types.

## IV. Indexing & Slicing

* **Basic Indexing:** Accessing single elements using integer indices (0-based).
* **Slicing:** Extracting subarrays using the colon `:` notation (`start:stop:step`). Works across multiple dimensions.
* **Integer Array Indexing (Fancy Indexing):** Selecting elements using arrays of indices. Allows picking arbitrary elements.
* **Boolean Array Indexing (Masking):** Selecting elements based on a boolean condition (creates a boolean mask). Very powerful for conditional selection.
* **Combining Indexing Types:** Mixing integer, slice, and boolean indexing.
* **Views vs. Copies:** Understanding when slicing creates a view (shares memory with the original) vs. a copy (independent). The importance of `ndarray.copy()`.

## V. Array Operations & Universal Functions (ufuncs)

* **Vectorization:** Performing operations on entire arrays without explicit Python loops (leveraging optimized C code).
* **Element-wise Operations:** Standard arithmetic (`+`, `-`, `*`, `/`, `//`, `%`, `**`) applied element by element.
* **Comparison Operations:** Element-wise comparisons (`==`, `!=`, `<`, `>`, `<=`, `>=`) resulting in boolean arrays.
* **Logical Operations:** `np.logical_and()`, `np.logical_or()`, `np.logical_not()` for combining boolean arrays.
* **Universal Functions (ufuncs):** Functions that operate element-wise on `ndarrays`.
    * Arithmetic: `np.add`, `np.subtract`, `np.multiply`, `np.divide`, `np.power`, `np.mod`, etc.
    * Trigonometric: `np.sin`, `np.cos`, `np.tan`, etc.
    * Exponential/Logarithmic: `np.exp`, `np.log`, `np.log10`, `np.sqrt`.
    * Rounding: `np.round`, `np.floor`, `np.ceil`.
* **Aggregation Functions:** Functions that summarize array data.
    * `np.sum()`, `np.prod()`
    * `np.mean()`, `np.median()`, `np.std()`, `np.var()`
    * `np.min()`, `np.max()`
    * `np.argmin()`, `np.argmax()`: Indices of the minimum/maximum values.
    * `np.cumsum()`, `np.cumprod()`: Cumulative sum/product.
* **Axis Parameter:** Performing aggregations along specific dimensions (rows, columns).

## VI. Broadcasting

* **Concept:** How NumPy handles operations between arrays of different, but compatible, shapes.
* **Broadcasting Rules:** The set of rules defining compatibility and how shapes are stretched.
* **Examples:** Operations between scalars and arrays, 1D and 2D arrays, etc.

## VII. Array Manipulation

* **Reshaping:**
    * `np.reshape()`, `ndarray.reshape()`: Changing array shape without changing data.
    * `np.ravel()`, `ndarray.flatten()`: Collapsing array to 1D (*Note: `ravel` may return a view, `flatten` always returns a copy*).
* **Transposing:** `np.transpose()`, `ndarray.T`: Permuting array dimensions (e.g., rows become columns).
* **Changing Dimensions:**
    * `np.expand_dims()`, `np.squeeze()`: Adding or removing dimensions of size 1.
    * Using `np.newaxis` or `None` during indexing to add dimensions.
* **Joining Arrays:**
    * `np.concatenate()`: Joining arrays along an existing axis.
    * `np.vstack()`: Stacking arrays vertically (row-wise).
    * `np.hstack()`: Stacking arrays horizontally (column-wise).
* **Splitting Arrays:**
    * `np.split()`: Splitting an array into multiple sub-arrays.
    * `np.vsplit()`, `np.hsplit()`: Splitting vertically/horizontally.
* **Adding/Removing Elements:** `np.append()`, `np.insert()`, `np.delete()` (*Note: These often return copies and can be inefficient for large arrays compared to pre-allocation*).
* **Repeating Elements:**
    * `np.tile()`: Constructing an array by repeating a given array.
    * `np.repeat()`: Repeating elements of an array.

## VIII. Linear Algebra (`np.linalg`)

* **Core Operations:**
    * `np.dot()` / `@` operator: Matrix multiplication / dot product.
    * `np.linalg.inv()`: Matrix inverse.
    * `np.linalg.det()`: Matrix determinant.
    * `np.trace()`: Sum along diagonals.
* **Decompositions:**
    * `np.linalg.eig()`: Eigenvalues and eigenvectors.
    * `np.linalg.svd()`: Singular Value Decomposition.
* **Solving Equations & Norms:**
    * `np.linalg.solve()`: Solving linear systems Ax = b.
    * `np.linalg.norm()`: Vector or matrix norm.

## IX. File Input/Output

* **NumPy Binary Files (`.npy`, `.npz`):** Efficient storage.
    * `np.save()`: Save a single array.
    * `np.savez()`: Save multiple arrays into an uncompressed `.npz` archive.
    * `np.savez_compressed()`: Save multiple arrays into a compressed `.npz` archive.
    * `np.load()`: Load arrays from `.npy` or `.npz` files.
* **Text Files (`.csv`, `.txt`):** Less efficient for large data but human-readable.
    * `np.savetxt()`: Save an array to a text file.
    * `np.loadtxt()`: Load data from a text file. (*Note: Pandas `read_csv` is often more flexible*).

## X. Advanced Concepts

* **Structured Arrays:** Arrays whose elements are C-struct-like structures with named fields.
* **Masked Arrays (`np.ma`):** Arrays that can have missing or invalid entries.
* **Performance Considerations:** Vectorization best practices, memory layout (C vs. Fortran order), avoiding unnecessary copies.
* **Interoperability:** Using NumPy arrays with other libraries (`Pandas`, `SciPy`, `Matplotlib`, `Scikit-learn`).