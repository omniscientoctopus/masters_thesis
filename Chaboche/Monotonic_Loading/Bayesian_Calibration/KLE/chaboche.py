import numpy as np
import copy

# to read the KLE surrogate data
import json

# custom modules
import KarhunenLoeveExpansion as KLE
import PolynomialChaosExpansion as PCE
from ChabocheModel import chaboche_uniform_isoprob_monotonic_transform

def surrogate_evaluate(X_test):

    # Read JSON
    with open('chaboche_monotonic_strainrate_1.json', 'r') as f:
        trained_model_data = json.load(f)

    # Initialise surrogate
    KLE_chaboche_monotonic_surrogate = KLE.KLE_surrogate_evaluate(trained_model_data, chaboche_uniform_isoprob_monotonic_transform)

    # Evaluate surrogate
    Y_surrogate = KLE_chaboche_monotonic_surrogate.surrogate_evaluate(X_test)

    return Y_surrogate
