# Collect parameters for fitting
import data
import numpy as np
from main import initiate_models, run_models
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

percentile = 5
fname = "../results/patchsize.txt"

def get_size(patch):
    BR_cells = patch.BR_original + patch.BR
    positions = [cell.pos[0] for cell in BR_cells]
    upper = np.percentile(positions, 100 - .5*percentile)
    lower = np.percentile(positions, .5*percentile)
    return upper - lower

if __name__ == '__main__':
    artificial = False
    prob_range = (.01, .04)
    N = 6
    probs = np.linspace(*prob_range, N)

    errors = []
    for value in probs:
        print("Running model for seed probability ",value)
        these_params, _ = data.get_params()
        these_params['seed_prob'] = value

        # Initiate and run models
        models = initiate_models(these_params)
        run_models(models)

        # Assign data to patches
        patches = data.assign_data(models, artificial)

        error = 0
        for patch in patches:
            if len(patch.BR_original) != 0:
                model_size = get_size(patch)
                # TODO: Use also other diameters than the last one?
                error += abs(model_size - patch.size[-1])

        errors.append(error)
        np.savetxt(fname, errors)

    results = np.loadtxt(fname)
    plt.plot(results)
    plt.show()