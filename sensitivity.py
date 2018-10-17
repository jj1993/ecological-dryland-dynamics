import numpy as np
from SALib.sample import saltelli

import data
from fit import get_total_loss
from main import initiate_models, run_models

# Amount of measurements, recommended 200
fname = "../results/sensitivity.txt"
N = 40
measurements = 1

# Problem definition. Make sure it is the same as in the Sobol analysis!
second_order = False
vars = ['alpha', 'gamma', 'c_rr', 'c_bb', 'c_br', 'c_rb']
bounds = [
    (0, .5),
    (0, .25),
    (0, .05),
    (0, .1),
    (0, .2),
    (0, .2)
]
problem = {
    'num_vars': len(vars),
    'names': vars,
    'bounds': bounds
}

def get_average_biomass(models, t):
    RL, BR = [], []
    for model in models:
        for patch in model.patches:
            RL.append(patch.RL_model[t])
            BR.append(patch.BR_model[t])

    return np.mean(RL), np.mean(BR)


def evaluate_model(params):
    # Initiate and run models
    models = initiate_models(params)
    run_models(models)

    # Assign data to patches
    patches = data.assign_data(models)

    results = []
    for t in range(measurements):
        t += 1
        mean = get_average_biomass(models, t)
        loss = get_total_loss(patches, t)
        results += [*mean, *loss]

    return results

param_values = saltelli.sample(problem, N, calc_second_order=second_order)
D = np.zeros((param_values.shape[0], measurements*4))
for i, params in enumerate(param_values):
    print("Sensitivity run %i out of %i"%(i, len(param_values)))
    model_params, _ = data.get_params()
    for var, val in zip(vars, params):
        model_params[var] = val

    D[i] = evaluate_model(model_params)
    np.savetxt(fname, D) # in case anything goes wrong...