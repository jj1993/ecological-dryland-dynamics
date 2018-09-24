from collections import defaultdict
from itertools import product
import numpy as np
from scipy.ndimage import gaussian_filter

import data
from cell import Cell
from growth import get_FL

### Outdated typedict for old patch loading
# TYPEDICT = {
#     '0.0': 'measure',
#     '0.2': 'RL',
#     '0.6': 'BR',
#     '0.9': 'exclusion',
#     '1.0': 'soil'
# }

class Model(object):
    def __init__(self, plot, params_model, params_RL, params_BR, patch_shape, seasonality):
        self.params = params_model
        self.BR = params_BR
        self.RL = params_RL
        self.patch_shape = patch_shape
        self.seasonality = seasonality

        self.time = 0
        self.data = defaultdict(list)
        self.patch_data = defaultdict(list)

        # initiate grids
        self.height, self.width = params_model["height"], params_model["width"]
        self.grid = np.empty((self.width, self.height), dtype=Cell)
        self.vegetation = {'BR' : [], 'RL' : []}

        # initiate patches
        self.patches = {'RP': [], 'BP': [], 'MP': [], 'RG': [], 'BG': [], 'MG': []}
        for patch in plot:
            self.addCells(patch['x'], patch['y'], patch['type'], patch['id'], patch['data'])
        self.allVegetation = self.grid.flat[self.grid.flat != None]

        # initiate diffusion matrices
        self.RL_diff = np.zeros((self.width, self.height))
        self.BR_diff = np.zeros((self.width, self.height))

        # initiate local and global connectivity measures
        self.updateConnectivity()

        # # run startup procedure and collect initial data
        # self.startup()
        # self.collect_data()

    def step(self, day):
        '''
        Advance the model by one step.
        '''

        self.time = day
        self.allVegetation = self.vegetation['BR'] + self.vegetation['RL']
        self.diffuse_biomass()
        np.random.shuffle(self.allVegetation)
        for cell in self.allVegetation:
            cell.step_cell()
        self.updateConnectivity()

    def updateConnectivity(self):
        # Collect all flowlengths
        self.FL_diff = np.zeros((self.width, self.height))
        for cell in self.allVegetation:
            this_x, this_y = cell.pos
            cell.upslope = this_y
            for y in reversed(range(this_y)):
                upslopeCell = self.grid[this_x][y]
                if upslopeCell != None:
                    cell.upslope -= y
                    break
            self.FL_diff[cell.pos] = cell.upslope
        self.FL_diff = get_FL(self.FL_diff)

        # Diffuse flowlengths over 2 sigma = 3 cells
        self.FL_diff = gaussian_filter(self.FL_diff, sigma= 3 / 2)

        # Safe all flowlengths correctly
        self.FL = 0
        for cell in self.allVegetation:
            cell.FL = self.FL_diff[cell.pos]
            self.FL += cell.FL

    def collect_data_fit(self):
        """
        Collects data for comparison with data on measurement points
        """
        data = defaultdict(float)
        for cell in self.allVegetation:
            if cell.has_data and cell.cell_type[0] == 'R':
                label = cell.patch + cell.cell_type[0]
                data[label] += cell.biomass

        for label in data.keys():
            self.patch_data[label].append(data[label])

        self.data["patch_biomass"].append(data)

    def collect_data_vis(self):
        """
        Collects data for visualisation in every timestep
        """
        biom_R = np.mean([cell.biomass for cell in self.vegetation["RL"]])
        biom_B = np.mean([cell.biomass for cell in self.vegetation["BR"]])
        comp = np.mean([1 - cell.grow_comp / cell.biomass for cell in self.vegetation["BR"] + self.vegetation["RL"]])
        pos = np.mean([cell.grow_pos for cell in self.vegetation["BR"] + self.vegetation["RL"]])
        conn = np.mean([cell.grow_conn for cell in self.vegetation["BR"] + self.vegetation["RL"]])

        nr_cells = len([cell.grow_pos for cell in self.vegetation["BR"] + self.vegetation["RL"]])
        biom_R_std = np.std([cell.biomass for cell in self.vegetation["RL"]])
        biom_B_std = np.std([cell.biomass for cell in self.vegetation["BR"]])
        comp_std = np.std([1 - cell.grow_comp / cell.biomass for cell in self.vegetation["BR"] + self.vegetation["RL"]])
        pos_std = np.std([cell.grow_pos for cell in self.vegetation["BR"] + self.vegetation["RL"]])
        conn_std = np.std([cell.grow_conn for cell in self.vegetation["BR"] + self.vegetation["RL"]])
        percent = np.mean([cell.grow_percent for cell in self.vegetation["BR"] + self.vegetation["RL"]])

        self.data['biom_R'].append(biom_R)
        self.data['biom_B'].append(biom_B)
        self.data['comp'].append(comp)
        self.data['pos'].append(pos)
        self.data['conn'].append(conn)
        self.data['biom_R_std'].append(biom_R_std  / np.sqrt(nr_cells))
        self.data['biom_B_std'].append(biom_B_std  / np.sqrt(nr_cells))
        self.data['comp_std'].append(comp_std  / np.sqrt(nr_cells))
        self.data['pos_std'].append(pos_std  / np.sqrt(nr_cells))
        self.data['conn_std'].append(conn_std / np.sqrt(nr_cells))
        self.data['percent'].append(percent)
        # TODO: Collect mortality and reproduction

    def startup(self):
        # Startup procedure doesn't seem necessary in this model
        pass

    def diffuse_biomass(self):
        # Collect actual biomasses for both species
        for cell in self.vegetation['RL']:
            self.RL_diff[cell.pos] = cell.biomass
        for cell in self.vegetation['BR']:
            self.BR_diff[cell.pos] = cell.biomass

        # Diffuse biomasses over the grid
        cs = self.params["cell_size"]
        self.RL_diff = gaussian_filter(self.RL_diff, sigma=self.RL["biom_sigma"] / (2 * cs))
        self.BR_diff = gaussian_filter(self.BR_diff, sigma=self.BR["biom_sigma"] / (2 * cs))

    def addCells(self, x_ref, y_ref, patch_type, id, has_data):
        biomass_data = data.get_biomass()

        patch = self.patch_shape[patch_type]
        patch_cells = []
        for x_pert, y_pert, cell_type in patch:
            coor = x_ref + 3 * x_pert, y_ref + 3 * y_pert

            if cell_type == "RL":
                coor_ext = coor[0] + 1, coor[1] + 1
                biomass = None
                if has_data:
                    label = id + cell_type[0]
                    try:
                        biomass = biomass_data[label][0]
                    except:
                        biomass = self.RL["biom"]
                        print(label)
                new_cell = Cell(self, self.grid, coor_ext, cell_type, id, has_data, biomass)
                self.grid[coor_ext] = new_cell
                self.vegetation[cell_type].append(new_cell)
                patch_cells.append(new_cell)
            if cell_type == "BR":
                for ext in product(range(3), range(3)):
                    coor_ext = tuple(map(sum, zip(coor, ext)))
                    new_cell = Cell(self, self.grid, coor_ext, cell_type, id, has_data)
                    self.grid[coor_ext] = new_cell
                    self.vegetation[cell_type].append(new_cell)
                    patch_cells.append(new_cell)

        self.patches[patch_type].append(patch_cells)