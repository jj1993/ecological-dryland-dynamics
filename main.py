import csv
import json
import sys
from itertools import product

import matplotlib
import numpy as np

from growth import get_FL
from scipy import optimize

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from model import Model
from visualisation import Visualize

NR_PLOTS = 24
VIS_PLOT = 10

def binValues(plot):
    new_plot = []
    for cell in plot[:,:,0].flatten():
        new_plot.append(round(cell, 1))

    return np.array(new_plot).reshape((66, 33)).T

def test_plot():
    grid = np.ones((10,15))
    grid[1,8], grid[5,8], grid[3,10] = 0.2, 0.2, 0.2
    grid[3,8], grid[7,8], grid[5,10] = 0.6, 0.6, 0.6
    return grid

if __name__ == "__main__":
    try:
        run_type = sys.argv[1]
        run_types = ["v", "f"] # Visualisation
        if not run_type in run_types:
            raise ValueError("run_type '%s' not recognised in list %s." %(run_type, run_types))
    except:
        run_type = None
        print("\nRunning model without run_type.\n")

    # load parameters
    config_model = "config_model.json"
    params_model = json.load(open(config_model))
    params_model['FL_glob_max'] = params_model['width'] * sum([get_FL(y) for y in range(params_model['height'])])
    config_RL = "config_RL.json"
    params_RL = json.load(open(config_RL))
    config_BR = "config_BR.json"
    params_BR = json.load(open(config_BR))
    config_patch_shape = "patch_shape.json"
    patch_shape = json.load(open(config_patch_shape))

    ### set-up models and visualisation - OLD PATCH INFO, OUTDATED
    # models = []
    # for plot_nr in range(NR_PLOTS):
    #     fname = "../data/plot-pixels/plot%i.png"%(plot_nr + 1)
    #     plot = matplotlib.image.imread(fname)
    #     plot = binValues(plot)
    #     models.append(Model(plot, params_model, params_RL, params_BR))
    # visualisation = Visualize(models[VIS_PLOT - 1])

    ### set-up test model
    # config_model = "config_model_test.json"
    # params_model = json.load(open(config_model))
    # plot = test_plot()
    # models = [Model(plot, params_model, params_RL, params_BR)]
    # visualisation = Visualize(models[0])

    # start visualisation

    # Read in patch data for every plot
    fname = "../data/Coordinates_final.txt"
    coordinates = csv.reader(open(fname), delimiter='\t')
    next(coordinates) # skips header row
    plots = [[] for n in range(NR_PLOTS)]
    for coor in coordinates:
        number, id, _, _, z, x, y = coor
        if id[-1] == '*':
            data = True
        else:
            data = False
        x = int(100 * float(x.replace(',', '.')) - 0.5) % 100
        y = int(100 * float(y.replace(',', '.')) - 0.5)
        z = float(z.replace(',', '.'))
        try:
            plot_nr = int(id[:2]) - 1
            patch_type = id[2:4]
        except:
            continue

        plots[plot_nr].append({
            'number' : number,
            'id' : id,
            'x' : x,
            'y' : y,
            'z' : z,
            'data' : data,
            'type' : patch_type
        })

    # set-up models and visualisation
    coordinates = list(product(range(params_model['width']), range(params_model['height'])))
    if run_type == 'v' or run_type == None:
        models = []
        for plot in plots:
            models.append(Model(plot, params_model, params_RL, params_BR, patch_shape, coordinates))

    if run_type == 'v':
        visualisation = Visualize(models[VIS_PLOT - 1], VIS_PLOT)
        visualisation.initiate()

    # run models
    if run_type == 'v' or run_type == None:
        for t in range(params_model["total_time"]):
            print("Timestep %i" % (t + 1))
            for model in models:
                model.step()

    if run_type == 'v':
        visualisation.update()
        # plt.pause(1)
        input("Press enter to close simulation...")
        visualisation.teardown()

    if run_type == 'f':
        timesteps = 10
        opti = 30
        x0 = (0.2, 0, 0.1, 0.1, 0.1, 0.1, 0.1)
        x_min = (0, 0, 0, 0, 0, 0, 0)
        x_max = (1, 1, 1, .25, .25, .25, .25)

        def getScore(models):
            score = 0
            for model in models:
                biom = model.data['biom_B'][-1]
                score += (biom - 30)**2
            return score

        def fun(test_params):
            print("Running new test...")
            alpha, beta, gamma, r_br, r_bb, r_rr, r_rb = test_params
            params_model["total_time"] = timesteps
            params_model["alpha"], params_model["beta"], params_model["gamma"] = alpha, beta, gamma
            params_BR["r_ir"], params_BR["r_ib"] = r_br, r_bb
            params_RL["r_ir"], params_RL["r_ib"] = r_rr, r_rb

            models = []
            for plot in plots:
                models.append(Model(plot, params_model, params_RL, params_BR, patch_shape, coordinates))

            for t in range(params_model["total_time"]):
                if t%10 == 0:
                    print("Timestep %i"%t)
                for model in models:
                    model.step()

            # TODO: assign score to model performance

            score = getScore(models)
            print(score, test_params)
            return score

        def print_fun(x, f, accepted):
            print("at minimum %.4f accepted %d" % (f, int(accepted)))


        class MyBounds(object):
            def __init__(self, x_min, x_max):
                self.xmin = np.array(x_min)
                self.xmax = np.array(x_max)

            def __call__(self, **kwargs):
                x = kwargs["x_new"]
                tmin = bool(np.all(x >= self.xmin))
                tmax = bool(np.all(x <= self.xmax))
                return tmin and tmax

        np.random.seed(1)
        mybounds = MyBounds(x_min, x_max)
        res = optimize.basinhopping(fun, x0, T=100, stepsize = 50, niter=5, callback=print_fun, accept_test=mybounds)#method='Nelder-Mead', options={'maxiter':50})
        print(res.x)