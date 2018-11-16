import copy
from itertools import combinations
from multiprocessing import Process

import numpy as np
import data
import subset
from main import initiate_models, run_models

import matplotlib
matplotlib.use('TkAgg')

def get_total_loss(patches, t):
    RL_diff, BR_diff = [], []
    for patch in patches:
        if patch.RL_data[t] > 0.0:
            RL_diff.append(np.abs((patch.RL_data[t] - patch.RL_model[t])) / patch.RL_data[t])
        if patch.BR_data[t] > 0.0:
            BR_diff.append(np.abs((patch.BR_data[t] - patch.BR_model[t])) / patch.BR_data[t])

    return np.nanmean(RL_diff), np.nanmean(BR_diff)

def get_subset_loss(subsets, t):
    RL_diff, BR_diff = [], []
    for set in subsets:
        RL_patch, BR_patch = [], []
        for patch_i, patch_j in combinations(set, end_date_nr):
            i_biom_R, i_biom_B = patch_i.RL_data[t], patch_i.BR_data[t]
            i_model_R, i_model_B = patch_i.RL_model[t], patch_i.BR_model[t]
            j_biom_R, j_biom_B = patch_j.RL_data[t], patch_j.BR_data[t]
            j_model_R, j_model_B = patch_j.RL_model[t], patch_j.BR_model[t]

            RL_patch.append(abs(
                (i_biom_R - j_biom_R) - #/(i_biom_R + j_biom_R) -
                (i_model_R - j_model_R) #/(i_model_R + j_model_R)
            ))
            BR_patch.append(abs(
                (i_biom_B - j_biom_B) - #/(i_biom_B + j_biom_B) -
                (i_model_B - j_model_B) #/(i_model_B + j_model_B)
            ))

        RL_diff.append(np.mean(RL_patch))
        BR_diff.append(np.mean(BR_patch))

    return np.nanmean(RL_diff), np.nanmean(BR_diff)

def get_diameter_loss(patches):
    t = 4
    loss = []
    for patch in patches:
        if patch.diam[t] > 0.0:
            loss.append(np.abs((patch.diam[t] - patch.diam_data[t])))

    return np.nanmean(loss)

def minimize_diam(param, param_range, params_model, iterations, artificial, fname):
    loss_list = []
    values = np.linspace(*param_range, iterations)
    for value in values:
        print("%s value: %.3f" % (param, value))

        # Collect parameters for fitting
        these_params = copy.deepcopy(params_model)
        these_params[param] = value

        # Initiate and run models
        models = initiate_models(these_params)
        run_models(models)

        # Assign data to patches
        patches = data.assign_data(models, artificial)

        # Calculate loss
        loss_list.append(get_diameter_loss(patches))

    np.savetxt(fname, loss_list)
    return

def minimize_total(param, param_range, params_model, iterations, subset_nr, artificial, fname):
    loss_list = []
    optimalisation_list = []
    values = np.linspace(*param_range, iterations)
    for value in values:
        print("%s value: %.3f" % (param, value))

        # Collect parameters for fitting
        these_params = copy.deepcopy(params_model)
        these_params[param] = value

        # Initiate and run models
        models = initiate_models(these_params)
        run_models(models)

        # Assign data to patches
        patches = data.assign_data(models, artificial)

        # Subset patches for parameter fitting
        if subset_nr == 5: subset_nr = 4
        subsets = subset.all(patches)[subset_nr]

        loss_list.append((
            *get_total_loss(patches, 1), *get_subset_loss(subsets, 1),
            *get_total_loss(patches, 2), *get_subset_loss(subsets, 2),
            *get_total_loss(patches, 3), *get_subset_loss(subsets, 3)
        ))

    # total_loss = get_total_loss(patches, 2)
    # optimalisation_list.append(total_loss)
    # opt_RL, opt_BR = zip(*optimalisation_list)
    # i = opt_RL.index(min(opt_RL))
    # j = opt_BR.index(min(opt_BR))
    #
    # if param in ['c_rr', 'c_rb']:
    #     params_model[param] = values[i]
    # elif param in ['c_bb', 'c_br']:
    #     params_model[param] = values[j]
    # else:
    #     params_model[param] = values[int(round((i+j)/2))]

    np.savetxt(fname, loss_list)
    return

if __name__ == '__main__':
    print('FITTING')

    artificial = False
    loc = '../results/'
    fname = input('What filename would you like to save the file under?: \n')
    comp_params = ['c_rr', 'c_bb', 'c_br', 'c_rb']
    comp_ranges = [(0, .4), (0, .3), (0, 1), (0, 1)]
    glob_params = ['alpha', 'beta', 'gamma']
    glob_ranges = [(0, .4), (0, .3), (0, 1)]
    seed_range = (0, .15)
    resolution = 10

    params_model, _ = data.get_params()

    proc = []
    for n in range(5):
        for i, (param, param_range) in enumerate(zip(comp_params, comp_ranges)):
            this_fname = "%s%s_%i_%s"%(loc, fname, n + 1, param)
            p = Process(target=minimize_total, args=(param, param_range, params_model, resolution, i + 2, artificial,
                                                     this_fname))
            p.start()
            proc.append(p)
    for p in proc:
        p.join()

    proc = []
    for n in range(5):
        for i, (param, param_range) in enumerate(zip(glob_params, glob_ranges)):
            this_fname = "%s%s_%i_%s"%(loc, fname, n + 1, param)
            p = Process(target=minimize_total, args=(param, param_range, params_model, resolution, i, artificial,
                                                     this_fname))
            p.start()
            proc.append(p)
    for p in proc:
        p.join()

    proc = []
    param, param_range = 'seed_prob', seed_range
    for n in range(10):
        this_fname = "%s%s_%i_%s"%(loc, fname, n + 1, param)
        p = Process(target=minimize_diam, args=(param, param_range, params_model, resolution, artificial,
                                                 this_fname))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()