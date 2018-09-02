import csv
import json

import matplotlib
import numpy as np
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
    # load parameters
    config_model = "config_model.json"
    params_model = json.load(open(config_model))
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
    models = []
    for plot in plots:
        models.append(Model(plot, params_model, params_RL, params_BR, patch_shape))
    visualisation = Visualize(models[VIS_PLOT - 1], VIS_PLOT)

    visualisation.initiate()

    # run models
    for t in range(params_model["total_time"]):
        print("Timestep %i" % (t + 1))
        for model in models:
            model.step()
    visualisation.update()

    # plt.pause(1)
    input("Press enter to close simulation...")
    visualisation.teardown()