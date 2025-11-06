# ShallowRandomization



# Quantum Rotation and Measurement Toolkit

This repository provides a collection of Python files and jupyter notebooks that allow the user to implement classical shadows protocols for Z2 Lattice Gauge theories utilizing a duality with the Ising model as detailed in the following work: TODO INSERT LINK
The code is organized into two main components: a `src/` directory containing the core source code, and an `example_notebooks/` directory demonstrating how to use the tools in practice.

---

## 📁 Repository Structure

## 🧩 `src/` – Source Code Modules

| File | Description |
|------|--------------|
| **`basis.py`** | Defines basis states satisfying Gauss' law and utilities for constructing and manipulating them. |
| **`measurements.py`** | Functions for computing expectation values and measurement statistics. Runs the experiments and creates shadow tables.|
| **`rotations.py`** | Implements general SU(2) and multi-qubit rotation operators. |
| **`rotations_ising.py`** | Specialized routines for post-processing on the Ising side of the duality. |
| **`rotation_tools.py`** | Helper functions for creating rotations defined in terms of operator on the Gauge theory side. |
| **`haar_tools.py`** | Utilities for generating Haar-random unitaries and random state vectors. |
| **`allpairs.py`** | Tools for reconstructing observables given a shadow table. compute_sample_protocol(args) takes in a shadow table and relevant arguments and computes a particular observable|
| **`utils.py`** | General-purpose helper functions shared across the codebase. |

---

## 📓 `example_notebooks/` – Jupyter Notebooks

| Notebook | Description |
|-----------|--------------|
| **`example-dual-product.ipynb`** | Runs the dual product protocol and calculate loop and string type observables|
| **`example-global-pairs.ipynb`** | Runs the Global Pairs protocol and estimates loop and string type observables|
| **`example-global-pairs-fbc.ipynb`** | Runs the Global Pairs protocol for a lattice with fixed boundary conditions and estimates loop and string type observables|

---

## ⚙️ Installation

Clone the repository and install dependencies (if any):

```bash
git clone https://github.com/frolandh/LGTShallowShadows.git
cd LGTShallowShadows/
pip install -r requirements.txt
```

---

## ✨ Authors

- Henry Froland, Jacob Bringewatt, Andreas Elben, Niklas Mueller

---

## 📄 License

This project is licensed under the **BSD 3-Clause License**.  
See the [LICENSE](LICENSE) file for full details.
