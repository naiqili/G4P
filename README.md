# G4P
Code for the paper Portfolio Selection with Graph-aware Gaussian Process Regression

## Prerequests

our implementation is mainly based on following packages:

```
python 3.7
pip install gpuinfo
pip install universal-portfolios
pip install tensorflow-gpu==1.15
pip install gpflow==1.5
```

Besides, some basic packages like `numpy` are also needed.

## Usage

Take the NYSE (N) dataset for example.

1. Run preprocess_NYSE_N.ipynb
2. Run run_NYSE_N.sh
3. Run strategy-NYSE_N.ipynb