import datetime
import math
from collections import defaultdict
from itertools import product
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

import data
from vegetation import Cell, RL_cell, BR_cell, Patch
from growth import get_FL

RUNOFF_RETURN = {
    5: 17, 7: 8, 9: 16, 11: 18, 12: 24, 13: 3, 14: 1, 15: 6, 19: 4, 20: 10, 21: 2, 22: 23
}
FL_DIFF = 3

# Calculating factor for total FL calculation
A = np.zeros((7, 7))
A[3, 3] = 1
A = gaussian_filter(A, sigma=FL_DIFF / 2)
FL_FACT = np.sum(A[3, :])

class Model(object):
    def __init__(self, nr, plot, params_model, patch_shape, seasonalities, cover_data):
        self.nr = nr
        if nr in RUNOFF_RETURN.keys():
            self.runoff_return = True
            params_model["beta"] = 0
        else:
            self.runoff_return = False
        self.params = params_model
        self.beta = params_model["beta"]
        self.patch_shape = patch_shape
        self.seasonal_RL, self.seasonal_BR = seasonalities
        self.time = data.start_date

        # initiate grids
        self.height, self.width = params_model["height"], params_model["width"]
        self.grid = np.empty((self.width, self.height), dtype=Cell)
        self.vegetation = {'BR' : [], 'RL' : []}
        self.data = defaultdict(list)
        self.cover_measurements = []
        self.cover_data = cover_data

        # initiate patches
        self.patches = []
        for patch in plot:
            self.addCells(patch['x'], patch['y'], patch['type'], patch['id'], patch['data'])
        self.allVegetation = self.grid.flat[self.grid.flat != None]

        # initiate local and global connectivity measures
        self.updateConnectivity()

    def step(self, visualize = False):
        '''
        Advance the model by one step.
        '''
        [BR_cell.clone() for BR_cell in self.vegetation['BR']]
        self.allVegetation = self.vegetation['BR'] + self.vegetation['RL']
        np.random.shuffle(self.allVegetation)

        self.diffuse_biomass()
        self.updateConnectivity()
        [cell.step_cell() for cell in self.allVegetation]

        # Update RL zone-of-influence
        [cell.biom_sigma_update() for cell in self.vegetation['RL']]

        if visualize:
            self.collect_data_vis()
        self.time += datetime.timedelta(1)

    def updateConnectivity(self):
        # Collect all flowlengths of individuals
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
        self.FL_diff = gaussian_filter(self.FL_diff, sigma= FL_DIFF / 2)

        # Collect all flowlengths for bottom row
        self.FL = 0
        for x, cell in enumerate(self.grid[:,-1]):
            if cell == None:
                for y in reversed(range(self.height)):
                    if self.grid[x, y] != None:
                        self.FL += FL_FACT * get_FL(self.height - y)
                        break
                else:
                    self.FL += FL_FACT * get_FL(self.height)

        # Save all flowlengths correctly and add individuals FL to total FL
        for cell in self.allVegetation:
            cell.FL = self.FL_diff[cell.pos]
            self.FL += cell.FL

    def collect_data_fit(self):
        """
        Collects data for comparison with data on measurement points
        """
        for patch in self.patches:
            patch.collect_data()

    def collect_data_vis(self):
        """
        Collects data for visualisation in every timestep
        """
        biom_R = np.mean([cell.biomass for cell in self.vegetation["RL"]])
        biom_B = np.mean([cell.biomass for cell in self.vegetation["BR"]])
        comp_RL = np.mean([cell.grow_comp for cell in self.vegetation['RL']])
        comp_BR = np.mean([cell.grow_comp for cell in self.vegetation['BR']])
        pos = np.mean([cell.grow_pos for cell in self.allVegetation])
        conn = np.mean([cell.grow_conn_loc for cell in self.allVegetation])

        biom_B_measured, biom_R_measured = [], []
        for patch in self.patches:
            if patch.has_data:
                if not 'R' in patch.type:
                    biom_B_measured.append(np.sum([cell.biomass for cell in patch.BR_original])/(9*patch.factor))
                if not 'B' in patch.type:
                    biom_R_measured.append(np.sum([cell.biomass for cell in patch.RL])/patch.factor)
        biom_B_measured = np.mean(biom_B_measured)
        biom_R_measured = np.mean(biom_R_measured)

        nr_cells = len([cell.grow_pos for cell in self.allVegetation])
        biom_R_std = np.std([cell.biomass for cell in self.allVegetation])
        biom_B_std = np.std([cell.biomass for patch in self.patches for cell in patch.BR])
        comp_RL_std = np.std([cell.grow_comp for cell in self.vegetation['RL']])
        comp_BR_std = np.std([cell.grow_comp for cell in self.vegetation['BR']])
        pos_std = np.std([cell.grow_pos for cell in self.allVegetation])
        conn_std = np.std([cell.grow_conn_loc for cell in self.allVegetation])

        self.data['biom_R'].append(biom_R)
        self.data['biom_B'].append(biom_B)
        self.data['biom_R_measured'].append(biom_R_measured)
        self.data['biom_B_measured'].append(biom_B_measured)
        self.data['comp_RL'].append(comp_RL)
        self.data['comp_BR'].append(comp_BR)
        self.data['pos'].append(pos)
        self.data['conn'].append(conn)
        self.data['biom_R_std'].append(biom_R_std  / np.sqrt(nr_cells))
        self.data['biom_B_std'].append(biom_B_std  / np.sqrt(nr_cells))
        self.data['comp_RL_std'].append(comp_RL_std  / np.sqrt(nr_cells))
        self.data['comp_BR_std'].append(comp_BR_std  / np.sqrt(nr_cells))
        self.data['pos_std'].append(pos_std  / np.sqrt(nr_cells))
        self.data['conn_std'].append(conn_std / np.sqrt(nr_cells))
        # TODO: Collect mortality and reproduction

    def diffuse_biomass(self):
        # initiate diffusion matrices
        self.RL_diff = np.zeros((self.width, self.height))
        self.BR_diff = np.zeros((self.width, self.height))

        # Collect actual biomasses for both species
        for cell in self.vegetation['RL']:
            self.RL_diff[cell.pos] = cell.biomass
        for cell in self.vegetation['BR']:
            self.BR_diff[cell.pos] = cell.biomass

        # Diffuse biomasses over the grid
        cs = self.params["cell_size"]
        self.RL_diff = diffuse(self.RL_diff, sigma=int(self.params["R_biom_sigma"] / cs))
        self.BR_diff = diffuse(self.BR_diff, sigma=int(self.params["B_biom_sigma"] / cs))

    def addCells(self, x_ref, y_ref, patch_type, patch_id, has_data):
        biomass_data, *_ = data.get_data()

        patch = self.patch_shape[patch_type]
        RL_cells, BR_cells = [], []
        for x_pert, y_pert, cell_type in patch:
            coor = x_ref + 3 * x_pert, y_ref + 3 * y_pert

            if cell_type == "RL":
                coor_ext = coor[0] + 1, coor[1] + 1
                biomass = None
                if has_data:
                    label = patch_id + cell_type[0]
                    if label in biomass_data.keys():
                        # Some plants have been removed from dataset or not been measured in reality
                        biomass = biomass_data[label][0]
                        if math.isnan(biomass):
                            biomass = self.params["R_biom"]
                    else:
                        has_data = False
                        biomass = self.params["R_biom"]
                new_cell = RL_cell(self, self.grid, coor_ext, patch_id, has_data, biomass)
                RL_cells.append(new_cell)

            if cell_type == "BR":
                for ext in product(range(3), range(3)):
                    coor_ext = tuple(map(sum, zip(coor, ext)))
                    new_cell = BR_cell(self, self.grid, coor_ext, patch_id, has_data)
                    BR_cells.append(new_cell)

        self.patches.append(Patch(self, patch_id, RL_cells, BR_cells, has_data))

    def collect_cover(self):
        # Calculate the relative cover of a modelled plot
        N = self.width * self.height
        RL_area = sum([10.11 * cell.biomass ** 0.5435 for cell in self.vegetation['RL']])
        BR_area = len([cell for cell in self.vegetation['BR']])

        self.cover_measurements.append((RL_area + BR_area) / N)

def diffuse(grid, sigma):
    size = sigma*2 + 1
    middle = sigma
    a = np.zeros((size,size))
    a[middle,middle] = 1.0
    mask = gaussian_filter(a, sigma=sigma/2)
    mask[middle,middle] = 0
    mask /= np.sum(mask)

    return cv2.filter2D(grid, -1, mask)