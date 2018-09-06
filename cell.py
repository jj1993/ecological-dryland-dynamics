import numpy as np
from numpy.random.mtrand import lognormal

from growth import f_pos, f_conn, f_comp

class Cell(object):
    def __init__(self, model, grid, pos, cell_type, patch = 0, has_data = False):
        self.model = model
        self.grid = grid
        self.patch = patch
        self.has_data = has_data
        self.age = 0
        self.FL = 0
        self.grow_pos, self.grow_comp, self.grow_conn, self.grow_percent = 1, 1, 1, 0

        assert isinstance(pos, tuple)
        self.pos = pos
        assert isinstance(cell_type, str)
        self.cell_type = cell_type
        if cell_type == "BR":
            self.params = {**self.model.params, **self.model.BR}
            self.biomass = self.params["biom"]
        elif cell_type == "RL":
            self.params = {**self.model.params, **self.model.RL}
            self.biomass = self.params["biom"]
        else:
            self.biomass = 0

    def step_cell(self):
        self.grow()
        if self.cell_type == "BR":
            self.clone()
        if self.biomass < 0:
            self.die()

    def grow(self):
        # Get right parameters
        p = self.params
        L_ext = 1 # seasonal/rain impact
        L = 2 # length of patches (m)
        alpha, beta, gamma = p["alpha"], p["beta"], p["gamma"]
        r_ir, r_ib, g, m, K, biom_sigma = p["r_ir"], p["r_ib"], p["g"], p["m"], p["K"], p["biom_sigma"]
        x, y = self.pos

        # Compute growth factor
        self.grow_pos = f_pos(y, alpha, L, self.model.params['cell_size'])
        self.grow_conn = f_conn(self.FL, self.model.FL, p)
        RL_eff, BR_eff = self.model.RL_diff[self.pos], self.model.BR_diff[self.pos]
        self.grow_comp = f_comp(RL_eff, BR_eff, r_ir, r_ib)
        r_i = self.grow_pos * self.grow_conn

        # Grow plant
        self.grow_percent = g * r_i * L_ext * (1 - self.biomass / K) - m
        dB_dt = self.biomass * self.grow_percent - self.grow_comp
        self.biomass += dB_dt

    def clone(self):
        prob = self.biomass * self.params["seed_prob"]

        if np.random.random() < prob:
            grid_size = self.params["cell_size"]
            angle = np.random.random() * 2 * np.pi
            mean, sigma = self.params["seed_mean"], self.params["seed_sigma"]
            dist = lognormal(mean, sigma) * mean

            x_disp = int(np.sin(angle) * dist / grid_size)
            y_disp = int(np.cos(angle) * dist / grid_size)
            x, y = self.pos
            new_x, new_y = x + x_disp, y + y_disp
            if new_x >= 0 and new_x < self.model.width and new_y >= 0 and new_y < self.model.height:
                new_pos = new_x, new_y
                if self.grid[new_pos] == None:
                    new_cell = Cell(self.model, self.grid, new_pos, "BR", patch = self.patch)
                    self.grid[new_pos] = new_cell
                    self.model.vegetation["BR"].append(new_cell)

    def make_BR_clone(self, parent):
        self.patch = parent.patch
        self.age = 0
        self.cell_type = "BR"
        self.params = {**self.model.params, **self.model.BR}
        self.biomass = self.params["biom"]

    def die(self):
        veg_list = self.model.vegetation[self.cell_type]
        for i, cell in enumerate(veg_list):
            if cell == self:
                del veg_list[i]
                break