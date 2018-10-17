import csv
import json
import pickle

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

NR_PLOTS = 24

date_format = "%d-%b-%y"
measurement_strings = [
    "08-Mar-17",
    "25-Apr-17",
    "04-Jul-17",
    "14-Feb-18",
    "24-May-18"
]
measurements = [datetime.strptime(date, date_format) for date in measurement_strings]
start_date = measurements[0]
end_date = measurements[-1]

def daterange():
    for n in range((end_date - start_date).days + 1):
        yield start_date + timedelta(n)

def get_timestep(day):
    return (day - start_date).days

def get_seasonality():
    fname = "../data/seasonal.csv"

    df = pd.read_csv(fname,delimiter=',', decimal='.')
    season_dict = dict(zip(
        [datetime.strptime(date, date_format) for date in df['Date']],
        df['NDVI/growth']
    ))

    # TODO: get complete seasonal data
    for date in daterange():
        if not date in season_dict:
            season_dict[date] = df['NDVI/growth'][0]

    return season_dict

def get_timestamps():
    measurements = []
    last_date = None
    for date in measurement_strings:
        new_date = datetime.strptime(date, date_format)
        if last_date:
            measurements.append((new_date - last_date).days)
        last_date = new_date

    return measurements

def get_data():
    fname = "../data/talud_data.csv"
    df = pd.read_csv(fname,delimiter=',', decimal=',')
    biomass_data = df.ix[:,['B1', 'B2', 'B3', 'B4', 'B5']].values
    patchsize_data = df.ix[:,['D_1', 'D_2', 'D_3', 'D_4.1', 'D_5.1']].values

    labels_data = df.ix[:,['plot', 'patch', 'specie']].values
    labels = ["%02d%s%s*%s" % (plot, patch[1:], patch[0], species[0]) for plot, patch, species in labels_data]
    exclusions = [(label, excl[0]) for label, excl in zip(labels, df.ix[:,['exclusion']].values)]
    exclusions = dict(set(exclusions))

    biom_dict = {}
    size_dict = {}
    for i, label in enumerate(labels):
        # Update biomass dict first
        b = biomass_data[i]
        # A Brachy individual contains 9 ramets, data gives biomass for only 1
        if label[-1] == 'B':
            b *= 9
        if label in biom_dict:
            for i, (d_old, d_new) in enumerate(zip(biom_dict[label], b)):
                if d_old == np.nan:
                    biom_dict[label][i] = d_new
                elif d_new == np.nan:
                    continue
                else:
                    biom_dict[label][i] = (d_old + d_new)/2
        else:
            biom_dict[label] = b

        # Update patchsize dict
        # TODO: how to treat patch sizes N1-N3? What to do with N4?
        s = patchsize_data[i]
        if label[-1] == 'B':
            size_dict[label] = s

    return biom_dict, size_dict, exclusions

def get_artificial_biomass():
    pickle_obj = open("../data/artificial_dataset.dict", "rb")
    data = pickle.load(pickle_obj)
    pickle_obj.close()

    return data

def get_params():
    # load parameters
    config_model = "config_model.json"
    params_model = json.load(open(config_model))
    params_model['FL_glob_max'] = 135021 # Value from parameter_boundaries file
    params_model['FL_loc_max'] = 2738 # Value from parameter_boundaries file
    config_patch_shape = "patch_shape.json"
    patch_shape = json.load(open(config_patch_shape))

    return params_model, patch_shape

def get_plots():
    fname = "../data/Coordinates_final.txt"
    coordinates = csv.reader(open(fname), delimiter='\t')
    next(coordinates)  # skips header row
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
            'number': number,
            'id': id,
            'x': x,
            'y': y,
            'z': z,
            'data': data,
            'type': patch_type
        })

    return plots

def assign_data(models, artificial = False):
    # Collcet data
    biom_data, size_data, exclusions = get_data()
    if artificial:
        biom_data = get_artificial_biomass()

    # Collect patch objects
    patch_objects = []
    for model in models:
        patch_objects += model.patches

    # Assign data to patch objects
    data_patches = []
    for label in biom_data:
        patch_id, cell_type = label[:-1], label[-1]
        for patch in patch_objects:
            if patch.id == patch_id:
                patch.add_data(cell_type, biom_data[label])
                if cell_type == 'B':
                    patch.add_size(size_data[label])
                # Check if patch has exclusion. Only use patches without exclusion
                if exclusions[label] == 'yes':
                    patch.exclusion = True
                else:
                    data_patches.append(patch)

    # Removing patches that were added twice (the patches from the mixed plots have two data entries in the biomass data!)
    data_patches = list(set(data_patches))

    return data_patches