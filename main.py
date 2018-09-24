import copy
import sys
from itertools import product

import matplotlib
import numpy as np

from scipy import optimize

import fit

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from model import Model
from visualisation import Visualize
import data

VIS_PLOT = 0
RUNOFF_RETURN = [5, 7, 9, 11, 12, 13, 14, 15, 19, 20, 21, 22]
params_model, params_RL, params_BR, patch_shape = data.get_params()
plots = data.get_plots()

def initiate_models():
    models = []
    for n, plot in enumerate(plots):
        these_params = copy.deepcopy(params_model)
        if (n + 1) in RUNOFF_RETURN:
            these_params['beta'] = 0.0
        models.append(Model(plot, these_params, params_RL, params_BR, patch_shape))

    return models

if __name__ == "__main__":
    try:
        run_type, vis_steps = sys.argv[1], int(sys.argv[2])
    except:
        run_type = None
        print("\nRunning model without run_type.\n")

    if run_type == 'v':
        # set-up models and visualisation
        models = initiate_models()
        visualisation = Visualize(models, VIS_PLOT)

        # run models
        for t in data.daterange():
            print("Timestep %i" % data.get_timestep(t))
            for model in models:
                model.step(t)
                model.collect_data_vis()
            if run_type == 'v' and data.get_timestep(t) % vis_steps == 0:
                visualisation.update()
                input("Press enter to continue simulation...")

        visualisation.teardown()

    else:
        # set-up models
        models = initiate_models()

        # run models
        for t in data.daterange():
            print("Timestep %i" % data.get_timestep(t))
            for model in models:
                model.step(t)

