# Hypotest

Statistical inference toolkit with three analysis modes:
- Frequentist A/B inference
- Bayesian A/B inference (PyMC)
- Survival analysis (Cox PH with RMST fallback)

## Quick Start

1. Create env and install:
```bash
pip install -e .[dev]
```

2. Run app:
```bash
streamlit run app.py
```

3. Run tests:
```bash
pytest
```

## Project Structure

- `app.py`: Streamlit UI
- `inference.py`: main orchestration
- `bayesiantest.py`: Bayesian A/B model
- `censoring.py`: survival analysis logic
- `validation.py`: input validation
- `tests/`: smoke/unit tests

## Production Notes

- Inputs are validated up-front in `run_inference`.
- Frequentist MI path runs tests across all imputations and combines p-values.
- Survival mode supports non-numeric group labels.
- Use pinned dependencies from `pyproject.toml`.
