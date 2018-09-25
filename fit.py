import copy
from operator import itemgetter
import numpy as np
import data
from model import Model

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

RUNOFF_RETURN = [5, 7, 9, 11, 12, 13, 14, 15, 19, 20, 21, 22]
params_model, params_RL, params_BR, patch_shape = data.get_params()
plots = data.get_plots()

def initiate_models():
    models = []
    seasonality = data.get_seasonality()

    for n, plot in enumerate(plots):
        these_params = copy.deepcopy(params_model)
        if (n + 1) in RUNOFF_RETURN:
            these_params['beta'] = 0.0
        models.append(Model(plot, these_params, params_RL, params_BR, patch_shape, seasonality))

    return models

def run_models():
    # set-up models
    models = initiate_models()

    for t in data.daterange():
        print("Timestep %i" % data.get_timestep(t))
        for model in models:
            if t in data.measurements:
                model.collect_data_fit()
            model.step(t)

    return models

def get_total_loss(biom_data, biom_model, labels, t=0):
    m = np.array(itemgetter(*labels)(biom_model))[:,t]
    d = np.array(itemgetter(*labels)(biom_data))[:,t]

    diff = m - d
    print(np.nanmean(m),np.nanmean(d))
    return np.nanmean(abs(diff))


def select_patches(patches_str, models):
    patches_obj = []
    for model in models:
        for p_model in model.patches:
            if p_model in patches_str:
                patches_obj.append(model.patches[p_model])

    return patches_obj


if __name__ == '__main__':
    print('FITTING')
    measurement_nr = 0
    patch_factor = {
            'RG': 6, 'BG': 6, 'MG': 3, 'RP': 2, 'BP': 2, 'MP': 1
        }

    biom_data = data.get_biomass()
    patches_str = biom_data.keys()
    # Multiply different patches by factors to correct for that not every individual was measured
    for label in patches_str:
        biom_data[label] *= patch_factor[label[2:4]]

    biom_model = {}
    models = run_models()
    for model in models:
        biom_model.update(model.patch_data)

    # # TODO: figure out how to deal with missing Rhamnus data
    # missing = ['05RP2*R', '10MP2*R', '17RP2*R', '18MP2*R']

    patches_obj = select_patches(patches_str, models)

    R_patches = [p for p in patches_str if p[-1] == 'R']
    R_error = get_total_loss(biom_data, biom_model, R_patches, measurement_nr)
    print("Rhamnus error: ", R_error)

    P_patches = [p for p in patches_str if p[-1] == 'B']
    P_error = get_total_loss(biom_data, biom_model, P_patches, measurement_nr)
    print("Brachy error: ", P_error)

    # def getScore(models):
    #     score = 0
    #     for model in models:
    #         biom = model.data['biom_B'][-1]
    #         score += (biom - 30) ** 2
    #     return score
    #
    #
    # def fun(test_params):
    #     print("Running new test...")
    #     alpha, beta, gamma, r_br, r_bb, r_rr, r_rb = test_params
    #     params_model["total_time"] = timesteps
    #     params_model["alpha"], params_model["beta"], params_model["gamma"] = alpha, beta, gamma
    #     params_BR["r_ir"], params_BR["r_ib"] = r_br, r_bb
    #     params_RL["r_ir"], params_RL["r_ib"] = r_rr, r_rb
    #
    #     models = []
    #     for plot in plots:
    #         models.append(Model(plot, params_model, params_RL, params_BR, patch_shape, coordinates))
    #
    #     for t in range(params_model["total_time"]):
    #         if t % 10 == 0:
    #             print("Timestep %i" % t)
    #         for model in models:
    #             model.step()
    #
    #     score = getScore(models)
    #     print(score, test_params)
    #     return score
    #
    #
    # def print_fun(x, f, accepted):
    #     print("at minimum %.4f accepted %d" % (f, int(accepted)))
    #
    #
    # class MyBounds(object):
    #     def __init__(self, x_min, x_max):
    #         self.xmin = np.array(x_min)
    #         self.xmax = np.array(x_max)
    #
    #     def __call__(self, **kwargs):
    #         x = kwargs["x_new"]
    #         tmin = bool(np.all(x >= self.xmin))
    #         tmax = bool(np.all(x <= self.xmax))
    #         return tmin and tmax
    #
    #
    # mybounds = MyBounds(x_min, x_max)
    # res = optimize.basinhopping(fun, x0, T=100, stepsize=50, niter=5, callback=print_fun,
    #                             accept_test=mybounds)  # method='Nelder-Mead', options={'maxiter':50})
    # print(res.x)