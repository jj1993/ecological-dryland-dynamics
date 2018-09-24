import copy
from collections import defaultdict

import numpy as np

import data
from model import Model

RUNOFF_RETURN = [5, 7, 9, 11, 12, 13, 14, 15, 19, 20, 21, 22]
FL_RESOLUTION = 2
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

def get_total_loss():
    pass


def alpha():
    # set-up models
    models = initiate_models()

    patch_types = ['RP', 'BP', 'MP', 'RG', 'BG', 'MG']
    patches_return = defaultdict(list)
    patches_no_return = defaultdict(list)
    for key in patch_types:
        for n, model in enumerate(models):
            if (n + 1) in RUNOFF_RETURN:
                patches_return[key] += model.patches[key]
            else:
                patches_no_return[key] += model.patches[key]

    sorted_patches = list(patches_return.values()) + list(patches_no_return.values())
    sorted_patches_new = []
    for category in sorted_patches:
        # Sorting patches on FL
        category.sort(key=lambda x: sum([cell.FL for cell in x]))
        # Splitting category in #FL_RESOLUTION equally sized groups of patches
        sorted_patches_new += np.array_split(category, FL_RESOLUTION)
    sorted_patches = sorted_patches_new

    # TODO: Only compare patches from same category

    # return alpha
    pass

def intra_comp():
    # set-up models
    models = initiate_models()

    # return r_rr, r_bb
    pass

def inter_comp():
    # set-up models
    models = initiate_models()

    # return r_rb, r_br
    pass

def gamma():
    # set-up models
    models = initiate_models()

    # return gamma
    pass

def beta():
    # set-up models
    models = initiate_models()

    # return beta
    pass

if __name__ == '__main__':
    print('FITTING')

    biom_data = data.get_biomass()
    biom_model = {}
    models = run_models()
    for model in models:
        biom_model.update(model.patch_data)

    print(biom_model)
    print()
    print(biom_data)

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