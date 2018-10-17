import numpy as np
from numpy.random.mtrand import lognormal
from growth import f_pos, f_conn, f_comp
from data import measurements

start_date = measurements[0]
patch_factor = {
    'RG': 6, 'BG': 6, 'MG': 3, 'RP': 2, 'BP': 2, 'MP': 1
}
test = 1

class Patch(object):
    def __init__(self, model, patch_id, RL_cells, BR_cells, has_data=False):
        self.model = model
        self.id = patch_id
        self.type = patch_id[2:4]
        self.RL = RL_cells
        self.BR = BR_cells
        self.BR_original = BR_cells
        self.has_data = has_data

        self.BR_data, self.RL_data = np.zeros(5), np.zeros(5)
        self.BR_model, self.RL_model = [], []
        self.factor = patch_factor[self.type]
        self.exclusion = False

        for cell in self.RL + self.BR:
            self.model.grid[cell.pos] = cell
            self.model.vegetation[cell.cell_type].append(cell)
            cell.patch = self

    def add_clone(self, cell):
        self.BR.append(cell)
        cell.patch = self
        self.model.vegetation["BR"].append(cell)
        self.model.grid[cell.pos] = cell

    def add_data(self, cell_type, biom):
        biom *= self.factor
        if cell_type == "B":
            self.BR_data = biom
        elif cell_type == "R":
            self.RL_data = biom
        else:
            raise ValueError

    def add_size(self, size):
        self.size = size

    def remove_cell(self, dead_cell):
        # Remove cell from patch
        if dead_cell.cell_type == "BR":
            for i, cell in enumerate(self.BR):
                if cell == dead_cell:
                    del self.BR[i]
                    break
        if dead_cell.cell_type == "RL":
            for i, cell in enumerate(self.RL):
                if cell == dead_cell:
                    del self.RL[i]
                    break

        # Remove cell from model list and grid
        self.model.grid[dead_cell.pos] = None
        for i, cell in enumerate(self.model.vegetation[dead_cell.cell_type]):
            if cell == dead_cell:
                del self.model.vegetation[dead_cell.cell_type][i]
                break

    def collect_biomass(self):
        if len(self.BR_original) > 0:
            self.BR_model.append(sum([cell.biomass for cell in self.BR_original]))
        else:
            self.BR_model.append(0.0)

        if len(self.RL) > 0:
            self.RL_model.append(sum([cell.biomass for cell in self.RL]))
        else:
            self.RL_model.append(0.0)

class Cell(object):
    def __init__(self, model, grid, pos, patch_id, has_data=False):
        self.model = model
        self.grid = grid
        self.id = patch_id
        self.has_data = has_data
        self.age = 0
        self.pos = pos
        self.max_biomass = 0

        # Load parameters for cell
        p = self.model.params
        self.cell_size = p["cell_size"]
        self.alpha, self.beta, self.gamma = p["alpha"], p["beta"], p["gamma"]
        self.FL_glob_max, self.FL_loc_max = p['FL_glob_max'], p['FL_loc_max']

    def grow(self):
        # Get parameters
        x, y = self.pos
        L_ext = self.model.seasonality[self.model.time] # seasonal impact
        RL_eff, BR_eff = self.model.RL_diff[self.pos], self.model.BR_diff[self.pos]

        # Compute growth factor
        self.grow_pos = f_pos(y, self.alpha, self.cell_size)
        self.grow_conn_glob, self.grow_conn_loc = f_conn(
            self.FL, self.model.FL, self.FL_glob_max, self.FL_loc_max, self.beta, self.gamma
        )
        self.grow_comp = f_comp(RL_eff, BR_eff, self.c_ir, self.c_ib)
        # if self.cell_type == 'BR':
        #     shape = 4
        #     fact = self.biomass * np.exp(-self.biomass * np.log(self.K * shape) / self.K) * shape
        #     self.grow_comp *= fact

        # Grow plant
        dynamics = self.grow_pos * self.grow_conn_glob * self.grow_conn_loc
        relative_growth = dynamics * (1 - self.biomass / self.K) - self.grow_comp
        dB_dt = self.biomass * self.g * (L_ext * relative_growth - self.m)
        self.biomass += dB_dt

    def die(self):
        self.max_biomass = max(self.biomass, self.max_biomass)
        if self.biomass < .25 * self.max_biomass:
            self.patch.remove_cell(self)

class BR_cell(Cell):
    def __init__(self, model, grid, pos, patch_id, has_data=False, biomass=None):
        super().__init__(model, grid, pos, patch_id, has_data)
        self.cell_type = "BR"
        p = self.model.params
        if self.model.time == start_date:
            self.biomass = p["B_biom"]
        else:
            self.biomass = p["B_a"]
        self.c_ir, self.c_ib = p["c_br"], p["c_bb"]
        self.g, self.m, self.K = p["B_g"], p["B_m"], p["B_K"]
        self.seed_prob, self.seed_mean, self.seed_sigma = p["seed_prob"], p["seed_mean"], p["seed_sigma"]

    def clone(self):
        prob = self.biomass * self.seed_prob

        if np.random.random() < prob:
            grid_size = self.cell_size
            angle = np.random.random() * 2 * np.pi
            mean, sigma = self.seed_mean, self.seed_sigma
            dist = lognormal(mean, sigma) * mean

            x_disp = int(np.sin(angle) * dist / grid_size)
            y_disp = int(np.cos(angle) * dist / grid_size)
            x, y = self.pos
            new_x, new_y = x + x_disp, y + y_disp
            if new_x >= 0 and new_x < self.model.width and new_y >= 0 and new_y < self.model.height:
                new_pos = new_x, new_y
                if self.grid[new_pos] == None:
                    new_cell = BR_cell(self.model, self.grid, new_pos, self.id)
                    self.patch.add_clone(new_cell)

    def step_cell(self):
        self.age += 1
        self.grow()
        self.clone()
        self.die()

class RL_cell(Cell):
    def __init__(self, model, grid, pos, patch_id, has_data=False, biomass=None):
        super().__init__(model, grid, pos, patch_id, has_data)
        self.cell_type = "RL"
        p = self.model.params
        if has_data:
            self.biomass = biomass
        else:
            self.biomass = p["R_biom"]
        self.c_ir, self.c_ib = p["c_rr"], p["c_rb"]
        self.g, self.m, self.K = p["R_g"], p["R_m"], p["R_K"]

    def step_cell(self):
        self.age += 1
        self.grow()
        self.die()