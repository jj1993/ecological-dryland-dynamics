import sys
from collections import defaultdict
import numpy as np
from model import Model
from visualisation import Visualize
import data

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

params_model, patch_shape = data.get_params()
plots = data.get_plots()

def initiate_models(params_model = params_model, season_fact = 1.0):
    models = []
    for n, plot in enumerate(plots):
        seasonality = data.get_seasonality()
        for key in seasonality.keys():
            seasonality[key] *= season_fact
        models.append(Model(n + 1, plot, params_model, patch_shape, seasonality))

    return models

def run_models(models):
    for t in data.daterange():
        for model in models:
            if t in data.measurements:
                model.collect_data_fit()
            model.step()
    return

def create_artificial_data():
    """
    Creates artifical dataset if called directly as main script
    """

    # get parameters
    p, _ = get_params()
    p["alpha"], p["gamma"] = .05, .15
    p["c_rr"], p["c_bb"] = .01, .05
    p["c_rb"], p["c_br"] = .15, .15


    print("Generating artificial dataset for parameters:")
    for key in p.keys():
        if key in ["alpha", "gamma", "c_rr", "c_bb", "c_br", "c_rb"]:
            print("%s: %.2f"%(key, p[key]))

    # Initiate and run models to obtain patches
    models = initiate_models(p)
    run_models(models)
    patches = []
    for model in models:
        patches += model.patches

    # Get data keys
    real_data, *_ = data.get_data()
    keys = real_data.keys()

    # Build up data dict from model data
    data_dict = defaultdict(np.array)
    for patch in patches:
        for key in keys:
            label = key[:-1]
            if label == patch.id:
                if key[-1] == "B":
                    data_dict[key] = np.array(patch.BR_model) / patch.factor
                if key[-1] == "R":
                    data_dict[key] = np.array(patch.RL_model) / patch.factor

    pickle_obj = open("../data/artificial_dataset.dict", "wb")
    pickle.dump(data_dict, pickle_obj)
    pickle_obj.close()


if __name__ == "__main__":
    try:
        run_type, vis_steps = sys.argv[1], int(sys.argv[2])
    except:
        run_type = None
        print("\nRunning model without run_type.\n")

    # set-up models
    models = initiate_models(params_model)

    if run_type == 'v':
        # set-up visualisation
        visualisation = Visualize(models)

        # run models
        for t in data.daterange():
            print("Timestep %i" % data.get_timestep(t))
            for model in models:
                model.step(visualize = True)
            if run_type == 'v' and data.get_timestep(t) % vis_steps == 0:
                visualisation.update()
                res = input("Press enter to continue simulation, type 'q' to stop current run:\n\t")
                if res == 'q':
                    break

        visualisation.teardown()

    else:
        # run models
        for t in data.daterange():
            print("Timestep %i" % data.get_timestep(t))
            for model in models:
                model.step()