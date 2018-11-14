from datetime import datetime
import pickle
import sys
from collections import defaultdict
from multiprocessing import Process, Manager

import numpy as np
from model import Model
from visualization import Visualize
import data

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

params_model, patch_shape = data.get_params()
plots = data.get_plots()

def initiate_models(params_model = params_model, double_seasonality = False, season_fact = 1.0):
    cover_data = data.get_covers()
    models = []
    for n, plot in enumerate(plots):
        seasonalities = data.get_seasonality(double_seasonality)
        for key in seasonalities[0].keys():
            seasonalities[0][key] *= season_fact
            seasonalities[1][key] *= season_fact
        models.append(Model(n + 1, plot, params_model, patch_shape, seasonalities, cover_data[n]))

    return models

def run_models(models):
    for t in data.daterange():
        for model in models:
            if t in data.measurements:
                model.collect_data_fit()
            if t in data.drone_dates:
                model.collect_cover()
            model.step()
    return

def create_artificial_data(params):
    """
    Creates artifical dataset if called directly as main script
    """

    print("Generating artificial dataset for parameters:")
    for key in p.keys():
        if key in ["alpha", "gamma", "c_rr", "c_bb", "c_br", "c_rb"]:
            print("%s: %.2f"%(key, p[key]))

    # Initiate and run models to obtain patches
    models = initiate_models(params)
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

def draw_plot(model):
    fig, ax = plt.subplots(figsize=(5,10))
    BR_grid = np.zeros((model.width, model.height))
    BR_grid += 0.00001
    for cell in model.vegetation["BR"]:
        if cell.biomass >= 0.01:  # Making sure seedlings don't distort picture
            BR_grid[cell.pos] = cell.biomass
    im1 = ax.imshow(BR_grid.T, cmap=plt.cm.Greens, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=0.5))
    # fig.colorbar(im1)

    RL_scatter = [(cell.pos, cell.biomass) for cell in model.vegetation["RL"]]
    for pos, biom in RL_scatter:
        ax.scatter(*pos, facecolors='none', s=80 * biom, edgecolors='r')

    loc = '../results/'
    plt.savefig("%splot_N5_model%i"%(loc, model.nr))
    plt.close()
    return

if __name__ == "__main__":
    try:
        run_type, vis_steps = sys.argv[1], int(sys.argv[2])
    except:
        run_type = None
        print("\nRunning model without run_type.\n")

    if run_type == 'a':
        """
        Run model for given set of parameters in order to validate fitting procedure
        """
        p, _ = data.get_params()
        p["alpha"], p["gamma"] = .15, .30
        p["c_rr"], p["c_bb"] = .01, .05
        p["c_rb"], p["c_br"] = .20, .15
        create_artificial_data(p)

    elif run_type == 'v':
        """
        Run interactive visualisation of model
        Second argument is the timestep-interval
        """
        # set-up models
        models = initiate_models(params_model)

        # set-up visualisation
        data.assign_data(models)
        visualisation = Visualize(models)

        # run models
        for t in data.daterange():
            print("Timestep %i" % data.get_timestep(t))
            for model in models:
                if t in data.measurements:
                    model.collect_data_fit()
                model.step(visualize = True)
            if data.get_timestep(t) % vis_steps == 0:
                visualisation.update()
                res = input("Press enter to continue simulation, type 'q' to stop current run:\n\t")
                if res == 'q':
                    break

        visualisation.teardown()

    elif run_type == 'i':
        """
        Generate images of all plots at the time the drone picture was made for comparison
        """
        # set-up models
        models = initiate_models(params_model)

        # Run model
        date_format = "%d-%b-%y"
        drone_date = datetime.strptime("08-Aug-17", date_format)
        for t in data.daterange():
            print("Timestep %i" % data.get_timestep(t))
            for model in models:
                model.step()

            if t == drone_date:
                # Save model end states as images
                for model in models:
                    draw_plot(model)
                break

    elif run_type == 's':
        """
        Run model under normal seasonality as well as two instead of one growth season
        Save results for analysis
        """
        def run(models):
            for t in data.daterange():
                for model in models:
                    model.step()
            return models

        def collect_data(i, return_dict):
            print("Starting run %i"%(i+1))

            # set-up models
            normal_models = initiate_models(params_model, double_seasonality = False)
            double_models = initiate_models(params_model, double_seasonality = True)
            normal_models = run(normal_models)
            double_models = run(double_models)

            return_dict[i] = []
            for models in (normal_models, double_models):
                data = [
                    np.nanmean([cell.biomass for model in models for cell in model.vegetation['BR']]),
                    np.nanmean([cell.biomass for model in models for cell in model.vegetation['RL']]),
                    np.nanmean([len(model.vegetation['BR'])/len(model.grid.flat) for model in models]),
                    np.nanmean([model.FL/params_model['FL_glob_max'] for model in models])
                ]
                return_dict[i] += data

            print("Finishing run %i"%(i+1))
            return

        proc = []
        return_dict = Manager().dict()
        for i in range(10):
            p = Process(target=collect_data, args=(i, return_dict))
            p.start()
            proc.append(p)
        for p in proc:
            p.join()

        return_list = []
        for key in return_dict.keys():
            return_list.append(return_dict[key])
        np.savetxt("../results/double_seasonal.txt", return_list)

    else:
        """
        Run model in plain-mode
        """
        # set-up models
        models = initiate_models(params_model)
        data.assign_data(models)
        run_models(models)

# TODO: Carrying capacity BR again!! Should be lower
#