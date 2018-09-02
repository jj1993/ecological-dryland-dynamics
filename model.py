from itertools import product
import numpy as np
from scipy.ndimage import gaussian_filter

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
    def __init__(self, plot, params_model, params_RL, params_BR, patch_shape):
        self.params = params_model
        self.BR = params_BR
        self.RL = params_RL
        self.patch_shape = patch_shape

        self.time = 0
        self.data = {
            'biomass' : [], 'biomass_std' : [],
            'pos' : [], 'pos_std' : [],
            'comp' : [], 'comp_std' : [],
            'conn' : [], 'conn_std' : [],
            'percent' : []
        }

        # initiate grids
        self.height, self.width = params_model["height"], params_model["width"]
        self.grid = np.empty((self.width, self.height), dtype=Cell)

        # initiate patches
        for patch in plot:
            self.addCells(patch['x'], patch['y'], patch['type'], patch['id'], patch['data'])

        # initiate soil cells
        self.coordinates = list(product(range(self.width), range(self.height)))
        for n, coor in enumerate(self.coordinates):
            if self.grid[coor] == None:
                new_cell = Cell(self, self.grid, coor, 'soil')
                self.grid[coor] = new_cell

        # initiate diffusion matrices
        self.RL_diff = np.zeros((self.width, self.height))
        self.BR_diff = np.zeros((self.width, self.height))

        # initiate local and global connectivity measures
        self.updateConnectivity()

        # # run startup procedure and collect initial data
        # self.startup()
        # self.collect_data()

    def step(self):
        '''
        Advance the model by one step.
        '''

        self.time += 1
        self.diffuse_biomass()
        for cell in self.grid.flatten():
            cell.step()
        self.updateConnectivity()
        self.collect_data()

    def updateConnectivity(self):
        # TODO: Check how much computational power this uses

        # Collect all flowlengths
        self.FL_diff = np.zeros((self.width, self.height))
        for cell in self.grid.flatten():
            if cell.cell_type == "BR" or cell.cell_type == "RL":
                this_x, this_y = cell.pos
                cell.upslope = this_y
                self.FL_diff[cell.pos] = get_FL(cell.upslope)
                for y in reversed(range(this_y)):
                    upslopeCell = self.grid[this_x][y].cell_type
                    if upslopeCell == "BR" or upslopeCell == "RL":
                        cell.upslope -= y
                        self.FL_diff[cell.pos] = get_FL(cell.upslope)
                        break

        # Diffuse flowlengths over 2 sigma = 3 cells
        self.FL_diff = gaussian_filter(self.FL_diff, sigma= 3 / 2)
        for cell in self.grid.flatten():
            if cell.cell_type != 'soil':
                cell.FL = self.FL_diff[cell.pos]

        # Compute the total flowlength of the plot
        self.FL = sum([cell.FL for cell in self.grid.flatten()])

    def collect_data(self):
        biomass = np.mean([cell.biomass for cell in self.grid.flatten() if cell.cell_type != 'soil'])
        comp = np.mean([1 - cell.grow_comp / cell.biomass for cell in self.grid.flatten() if cell.cell_type != 'soil'])
        pos = np.mean([cell.grow_pos for cell in self.grid.flatten() if cell.cell_type != 'soil'])
        conn = np.mean([cell.grow_conn for cell in self.grid.flatten() if cell.cell_type != 'soil'])

        nr_cells = len([cell.grow_pos for cell in self.grid.flatten() if cell.cell_type != 'soil'])
        biomass_std = np.std([cell.biomass for cell in self.grid.flatten()])
        comp_std = np.std([1 - cell.grow_comp / cell.biomass for cell in self.grid.flatten() if cell.cell_type != 'soil'])
        pos_std = np.std([cell.grow_pos for cell in self.grid.flatten() if cell.cell_type != 'soil'])
        conn_std = np.std([cell.grow_conn for cell in self.grid.flatten() if cell.cell_type != 'soil'])
        percent = np.mean([cell.grow_percent for cell in self.grid.flatten() if cell.cell_type != 'soil'])

        self.data['biomass'].append(biomass)
        self.data['comp'].append(comp)
        self.data['pos'].append(pos)
        self.data['conn'].append(conn)
        self.data['biomass_std'].append(biomass_std  / np.sqrt(nr_cells))
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
        for i, j in self.coordinates:
            cell = self.grid[i, j]
            if cell.cell_type == 'RL':
                self.RL_diff[i, j] = cell.biomass
            if cell.cell_type == 'BR':
                self.BR_diff[i, j] = cell.biomass

        # Diffuse biomasses over the grid
        cs = self.params["cell_size"]
        self.RL_diff = gaussian_filter(self.RL_diff, sigma=self.RL["biom_sigma"] / (2 * cs))
        self.BR_diff = gaussian_filter(self.BR_diff, sigma=self.BR["biom_sigma"] / (2 * cs))

    def addCells(self, x_ref, y_ref, patch_type, id, has_data):
        patch = self.patch_shape[patch_type]
        for x_pert, y_pert, cell_type in patch:
            coor = x_ref + x_pert, y_ref + y_pert
            new_cell = Cell(self, self.grid, coor, cell_type, id, has_data)
            self.grid[coor] = new_cell