# IQR-EPCLR: A Scalable and Deterministic High-Breakdown Regression for Large-Scale Data

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()

Official implementation and experiment scripts for the paper:
**"IQR-EPCLR: A Scalable and Deterministic High-Breakdown Regression for Large-Scale Data"**

---

## 🚀 Abstract

Existing high-breakdown-point regression estimators suffer from **prohibitive computational cost** and **stochasticity**, limiting their use on large-scale datasets.
We introduce **IQR-EPCLR (Interquartile Range-based Efficient Pairwise Candidate Line Ranking)**, a **deterministic** and **scalable** robust regression algorithm.

Key highlights:
- ✅ Achieves **maximum 50% breakdown point** via MM-estimator architecture
- ✅ Strong **local robustness** (proved via influence function)
- ✅ **Consistent** under mild assumptions
- ✅ **10–100× faster** than state-of-the-art robust methods on large data

---

## 📂 Repository Structure

```
iqr-epclr-project/
│
├── src/
│   ├── iqr_epclr.py                    # Core IQR-EPCLR implementation
│   ├── utils.py                        # Utility functions and data generation
│   └── benchmarks.py                   # Comparison with other methods
│
├── experiments/
│   ├── run_simulations.py              # Monte Carlo simulations
│   ├── run_nyc_taxi_analysis.py        # Large-scale NYC Taxi dataset analysis
│   └── run_benchmark.py                # Comprehensive method comparison
│
├── analysis/
│   ├── generate_figures_and_tables.py  # Generate paper-ready plots & tables
│   └── visualization.py                # Plotting utilities
│
├── data/
│   ├── download_data.py                # Dataset downloader & preprocessing
│   └── datasets/                       # Raw and processed datasets
│
├── tests/
│   ├── test_iqr_epclr.py              # Unit tests for core implementation
│   └── test_utils.py                   # Tests for utility functions
│
├── examples/
│   ├── quick_start.py                  # Basic usage example
│   ├── advanced_usage.py               # Advanced features demonstration
│   └── plotting_example.py             # Visualization examples
│
├── results/                            # (Generated) Experiment outputs
├── figures/                            # (Generated) Paper figures
├── requirements.txt                    # Dependencies
├── setup.py                           # Package installation
├── LICENSE                            # MIT License
└── README.md                          # Project documentation
```

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/your-username/iqr-epclr-project.git
cd iqr-epclr-project

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode (optional)
pip install -e .
```

## 📋 Requirements

```txt
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
pandas>=1.3.0
seaborn>=0.11.0
```

---

## 🚀 Quick Start

### Basic Usage

```python
from src.iqr_epclr import IQR_EPCLR
import numpy as np

# Load your data
X = np.random.randn(100, 2)
y = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(100) * 0.5

# Fit robust regression
model = IQR_EPCLR(max_iter=100, tol=1e-6)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"R² Score: {model.score(X, y):.4f}")
```

### Complete Example

See [`examples/quick_start.py`](examples/quick_start.py) for a complete working example with:
- Data generation with outliers
- Model fitting and prediction
- Comparison with OLS and other robust methods
- Visualization of results

```bash
python examples/quick_start.py
```

---

## 📊 Reproducing Paper Results

### Step 1: Download Datasets

```bash
# Download and preprocess all benchmark datasets
python data/download_data.py
```

This will download:
- NYC Taxi dataset (large-scale real data)
- Standard robust regression benchmarks (starsCYG, hbk, coleman)
- Synthetic datasets for controlled experiments

### Step 2: Run Experiments

**Monte Carlo Simulations (Table 1 in paper)**
```bash
python experiments/run_simulations.py
```

**Real Dataset Analysis (Table 2 in paper)**
```bash
python experiments/run_nyc_taxi_analysis.py
```

**Comprehensive Benchmarks (Figure 2-3 in paper)**
```bash
python experiments/run_benchmark.py
```

### Step 3: Generate Figures and Tables

```bash
python analysis/generate_figures_and_tables.py
```

Output:
- Figures saved to `figures/`
- Tables printed to console and saved to `results/`

---

## 🔬 Algorithm Overview

**IQR-EPCLR** combines three key innovations:

1. **IQR-based Scale Estimation**: Uses interquartile range for robust, deterministic scale estimation
2. **Efficient Pairwise Candidate Selection**: Smart sampling strategy reduces computational complexity
3. **MM-estimator Framework**: Iterative re-weighting ensures convergence and optimality

### Key Features

| Feature | IQR-EPCLR | LTS | S-estimator | MM-estimator |
|---------|-----------|-----|-------------|--------------|
| **Breakdown Point** | 50% | 50% | 50% | 37% |
| **Deterministic** | ✅ | ❌ | ❌ | ❌ |
| **Scalable** | ✅ | ❌ | ❌ | ✅ |
| **Time Complexity** | O(n log n) | O(n²) | O(n²) | O(n log n) |

---

## 📈 Performance Results

### Computational Efficiency
- **10-100× faster** than LTS regression on large datasets
- **Linear scaling** with sample size (vs quadratic for competitors)
- **Deterministic results** (no random initialization)

### Statistical Properties
- **Maximum breakdown point**: 50%
- **High efficiency**: 95% under normal errors
- **Consistency**: Proven under mild regularity conditions
- **Robustness**: Superior performance under heavy contamination

See detailed results in our paper and `results/` directory.

---

## 🧪 Testing

Run the test suite to verify installation:

```bash
# Run all tests
python -m pytest tests/

# Run specific test files
python -m pytest tests/test_iqr_epclr.py -v
python -m pytest tests/test_utils.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Manual Testing

```bash
# Test basic functionality
python tests/test_iqr_epclr.py

# Test on synthetic data
python examples/quick_start.py
```

---

## 📚 Documentation

### Core Classes

- **`IQR_EPCLR`**: Main robust regression estimator
- **`RobustScaler`**: Robust data preprocessing utilities
- **`BenchmarkSuite`**: Automated comparison framework

### Key Parameters

- `max_iter`: Maximum MM-estimator iterations (default: 100)
- `tol`: Convergence tolerance (default: 1e-6)
- `scale_est`: Scale estimator ('iqr' or 'mad', default: 'iqr')
- `breakdown_point`: Target breakdown point (default: 0.5)

See docstrings in source files for detailed parameter descriptions.

---

## 🔍 Advanced Usage

### Custom Scale Estimators

```python
# Using MAD instead of IQR
model = IQR_EPCLR(scale_est='mad')

# Custom breakdown point
model = IQR_EPCLR(breakdown_point=0.3)
```

### Integration with Scikit-learn

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', IQR_EPCLR())
])

pipeline.fit(X_train, y_train)
```

### Large Dataset Handling

```python
# For very large datasets, use chunking
model = IQR_EPCLR(max_iter=50)  # Reduced iterations for speed
model.fit(X_large, y_large)
```

See [`examples/advanced_usage.py`](examples/advanced_usage.py) for complete examples.

---

## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{iqr_epclr2025,
  title={IQR-EPCLR: A Scalable and Deterministic High-Breakdown Regression for Large-Scale Data},
  author={Your Name and Co-authors},
  journal={Journal of Computational Statistics},
  year={2025},
  volume={XX},
  pages={XXX--XXX},
  publisher={Springer},
  doi={10.1007/sxxxxx-xxx-xxxxx-x}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/ examples/
isort src/ tests/ examples/
```

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/your-username/iqr-epclr-project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/iqr-epclr-project/discussions)
- **Email**: [your-email@university.edu](mailto:your-email@university.edu)

## 🔗 Related Work

- [Robust Statistics Library](https://github.com/statsmodels/statsmodels)
- [Scikit-learn Robust Estimators](https://scikit-learn.org/stable/modules/linear_model.html#robustness-regression-outliers-and-modeling-errors)
- [LIBRA Toolbox](https://wis.kuleuven.be/stat/robust/LIBRA)

---

**Keywords:** robust regression, high-breakdown point, outlier detection, large-scale data, MM-estimator, computational statistics, machine learning
