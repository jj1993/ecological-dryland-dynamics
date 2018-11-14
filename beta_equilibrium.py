import pickle
from itertools import product
from multiprocessing import Process, Manager
import numpy as np
from main import initiate_models
from model import RUNOFF_RETURN
import data

def get_biom(model, cell_type):
    return sum([cell.biomass for cell in model.vegetation[cell_type]])

def evaluate_model(model, settings, cell_type):
    key = "%i%s" % (model.nr, cell_type)

    # Collect cover percentage
    model.collect_cover()
    beta, diff, _ = settings[key]
    settings[key] = (beta, diff, model.cover_measurements[0])
    before = get_biom(model, cell_type)

    # Set beta and run model
    model.beta = beta
    [model.step() for t in data.daterange()]

    # Evaluate cover development
    after = get_biom(model, cell_type)
    beta, _, cover = settings[key]
    settings[key] = (beta, after - before, cover)
    return

if __name__ == "__main__":
    loc = "../results/"
    fname = "beta_equilibrium"
    init_beta = .5
    res = 5
    cell_types = ["BR", "RL"]

    # Initiate dictionary to keep track of results
    keys = ["%i%s" % (i, cell_type) for cell_type in cell_types for i in RUNOFF_RETURN.values()]
    for alpha_fact, gamma_fact in product([0,1,2], repeat=2):
        settings = Manager().dict()
        for key in keys:
            settings[key] = (init_beta, 0, 0)

        for b in range(res):
            print("Simulation step %i"%(b+1))
            # Update beta settings
            step_size = init_beta * 0.5 ** b
            for key in settings.keys():
                old_beta, diff, cover = settings[key]
                new_beta = old_beta + np.sign(diff) * step_size
                settings[key] = (new_beta, diff, cover)

            # Set up procedure for both species for all plots
            proc = []
            for cell_type in cell_types:
                # Initiate models without runoff return
                params, _ = data.get_params()
                params["alpha"] *= alpha_fact
                params["gamma"] *= gamma_fact
                models = [model for model in initiate_models(params) if not model.runoff_return]

                # Run models and collect differences
                for model in models:
                    p = Process(target=evaluate_model, args=(model, settings, cell_type))
                    p.start()
                    proc.append(p)

            # Run procedures in multi-processing
            for p in proc:
                p.join()
                proc = []

        # Save results for analysis
        D = [settings[key] for key in keys]
        name = "%s%s_alpha%i_gamma%i.txt"%(loc, fname, alpha_fact, gamma_fact)
        np.savetxt(name, D)