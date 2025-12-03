# Bovine Behavior Classification

Feature engineering workflow for Japanese Black Beef Cow accelerometer data.

## Current workflow

1. Load `cow1.csv` to `cow6.csv`
2. Remove unlabeled rows
3. Split each cow time series into sliding windows
4. Extract statistical and FFT features from `AccX`, `AccY`, and `AccZ`

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python app/app.py
```
