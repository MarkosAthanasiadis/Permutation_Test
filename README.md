# Significance Testing Framework

## Overview

The **Significance Testing Framework** is a Python-based tool designed for permutation testing on datasets. It provides functionality to assess the statistical significance of machine learning models' performance by comparing against shuffled label baselines.

### Features
- Flexible input handling for different dataset configurations.
- Parallelized computation using all available CPU cores.
- Modular design for easy customization and integration.
- Detailed results, including:
  - CCR values for real and shuffled datasets.
  - Statistical metrics such as p-values and z-scores.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/significance-testing.git
   cd significance-testing
   ```

2. Install dependencies:
   
   You can choose between using `pip` or `conda` for dependency management:

   #### Using `pip`:
   If you prefer `pip`, use the `requirements.txt` file:
   ```bash
   pip install -r requirements_ma.txt
   ```

   #### Using `conda`:
   If you prefer `conda`, use the provided environment file to create a new environment:
   ```bash
   conda env create -f env_ma.yml

---

## Usage

1. **Prepare Input Parameters**:
   Create a parameter file (see `parameters.yml`) with the suggested format.

Note: Ensure your dataset (e.g., hello_world.pkl) is a valid Python pickle file containing a dictionary with keys **data** and **labels** that match the expected format (see hello_world.pkl).

2. **Run the Permutation Test**:
   Execute the test using the `run_test.py` script:
   ```bash
   python run_test.py
   ```
   You will be prompted to enter the parameter file name (parameters.yml).

3. **Results**:
   The results will be saved in the specified `main path` under a `data_info/` directory. Results include:
   - Pickle file with CCR metrics.
   - Statistical significance values (p-values, z-scores).

---

## Project Structure

```
permutation-test/
├── significance/
│   ├── __init__.py
│   ├── run_test.py        # Main script for running the permutation test
│   ├── main.py            # Core computation logic
│   ├── utils/             # Helper modules
│       ├── shuffle_labels.py
│       ├── model_functions.py
│       ├── pvalue.py
│       ├── data_loader.py
│       └── timer.py
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
├── LICENSE                # License for the project
└── setup.py               # Installation script
```

---

## Contributing

We welcome contributions to improve this framework. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the [MIT License](LICENSE).
