"""Forecasting models. Every model follows the same protocol:

1. choose hyperparameters on (train -> validation) only;
2. refit on train + validation (the first 80%) with those hyperparameters;
3. predict the test set once.

Scalers are always fitted on the rows the model is fitted on, never on rows it predicts.
"""
