import csv
import json
from collections import defaultdict

from growth import get_FL
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
end_date = measurements[0]

def daterange():
    for n in range((end_date - start_date).days + 1):
        yield start_date + timedelta(n)

def get_timestep(day):
    return (day - start_date).days

def get_seasonality():
    fname = "../data/seasonal.csv"

    df = pd.read_csv(fname,delimiter=',', decimal=',')
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

def get_biomass():
    fname = "../data/biom_data.csv"
    df = pd.read_csv(fname,delimiter=',', decimal=',')
    biomass_data = df.ix[:,['B1', 'B2', 'B3', 'B4', 'B5']].values

    labels_data = df.ix[:,['plot', 'patch', 'specie']].values
    labels = ["%02d%s%s*%s" % (plot, patch[1:], patch[0], species[0]) for plot, patch, species in labels_data]

    data = {}
    for i, label in enumerate(labels):
        # A Brachy individual contains 9 ramets, data gives biomass for only 1
        b = biomass_data[i]
        if label[-1] == 'B':
            b *= 9
        if label in data:
            for i, (d_old, d_new) in enumerate(zip(data[label], b)):
                if d_old == np.nan:
                    data[label][i] = d_new
                elif d_new == np.nan:
                    continue
                else:
                    data[label][i] = (d_old + d_new)/2
        else:
            data[label] = b

    return data

def get_params():
    # load parameters
    config_model = "config_model.json"
    params_model = json.load(open(config_model))
    # params_model['FL_glob_max'] = params_model['width'] * sum([get_FL(y) for y in range(params_model['height'])])
    params_model['FL_glob_max'] = 135021 # Value from parameter_boundaries file
    params_model['FL_loc_max'] = 2738 # Value from parameter_boundaries file
    config_RL = "config_RL.json"
    params_RL = json.load(open(config_RL))
    config_BR = "config_BR.json"
    params_BR = json.load(open(config_BR))
    config_patch_shape = "patch_shape.json"
    patch_shape = json.load(open(config_patch_shape))

    return params_model, params_RL, params_BR, patch_shape

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

def test_plot():
    grid = np.ones((10,15))
    grid[1,8], grid[5,8], grid[3,10] = 0.2, 0.2, 0.2
    grid[3,8], grid[7,8], grid[5,10] = 0.6, 0.6, 0.6
    return grid