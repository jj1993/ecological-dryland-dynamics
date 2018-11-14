import numpy as np
from SALib.sample import saltelli

import data
from fit import get_total_loss
from main import initiate_models, run_models
from multiprocessing import Process, Manager

# Amount of measurements, recommended 200
fname = "../results/sensitivity.txt"
N = 40
measurements = 4

# Problem definition. Make sure it is the same as in the Sobol analysis!
second_order = True
vars = ['alpha', 'gamma', 'c_rr', 'c_bb', 'c_br', 'c_rb', 'seed_prob']
bounds = [
    (0, 1),
    (0, 1),
    (0, .15),
    (0, .15),
    (0, 1),
    (0, 1),
    (0, .03)
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


def evaluate_model(params, i, D, return_dict):
    print("Starting sensitivity run %i out of %i"%(i + 1, len(D)))

    # Defining model parameters
    model_params, _ = data.get_params()
    for var, val in zip(vars, params):
        model_params[var] = val

    # Initiate and run models
    models = initiate_models(model_params)
    run_models(models)

    # Assign data to patches
    patches = data.assign_data(models)

    results = []
    for t in range(measurements):
        t += 1
        mean = get_average_biomass(models, t)
        loss = get_total_loss(patches, t)
        results += [*mean, *loss]

    return_dict[i] = results

    print("Finishing sensitivity run %i out of %i"%(i + 1, len(D)))
    return

if __name__ == '__main__':
    # Define parameter settings and empty results dataset
    param_values = saltelli.sample(problem, N, calc_second_order=second_order)
    D = np.zeros((param_values.shape[0], measurements*4))

    # Run the model in parallel
    return_dict = Manager().dict()
    proc = []
    for i, params in enumerate(param_values):
        p = Process(target=evaluate_model, args=(params, i, D, return_dict))
        p.start()
        proc.append(p)

        if (i%40 == 0 and i != 0) or i == len(param_values) - 1:
            for p in proc:
                p.join()
                proc = []

            for i in range(len(return_dict.keys())):
                D[i] = return_dict[i]

    # Save results for analysis
    np.savetxt(fname, D)