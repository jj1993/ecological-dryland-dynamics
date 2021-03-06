from collections import defaultdict

import matplotlib
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider, Button, RadioButtons
import scipy.interpolate as inter

import data
from data import start_date

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

class Visualize(object):
    def __init__(self, models):
        self.params = data.get_params()[0]
        self.models = models
        self.model = models[0]
        self.nr = 1

        # Starting interactive matplotlib plot
        plt.ion()
        self.fig = plt.figure(figsize=(15, 10))

        # Dividing matplotlib figure into three sections
        gs = GridSpec(4, 3)
        self.ax1 = plt.subplot(gs[:, :-1])
        # self.ax1.yaxis.set_label_position("right")
        self.ax2 = plt.subplot(gs[-2:, -1])
        self.ax3 = plt.subplot(gs[:-2, -1])

        # Make grid, colorbar and text on grid
        self.cax1 = make_axes_locatable(self.ax1).append_axes("right", size="3%", pad="2%")
        self.fig.canvas.mpl_connect('pick_event', self.onpick)
        # self.gridtext = [
        #     self.ax1.text(i, j, int(self.model.grid[i, j].upslope), color='white', ha='center', va='center')
        #     for i, j in zip(*self.grid.nonzero())
        # ]

        # Define an action for modifying the line when any slider's value changes
        def sliders_on_changed(val):
            self.nr = int(val)
            self.model = self.models[self.nr - 1]
            self.update()

        # Define an axes area and draw a slider in it
        self.slider_ax = self.fig.add_axes([0.23, 0.03, 0.67, 0.03])#, axisbg='blue')
        self.slider = Slider(self.slider_ax, 'Plot', 1, 24, valinit=self.nr, valfmt='%i')
        self.slider.on_changed(sliders_on_changed)

        def change_param(param):
            def fun(val):
                self.params[param] = val
                for model in models:
                    model.params = self.params
                    for cell in model.allVegetation:
                        cell.update_params()
            return fun

        # Define an axes area and draw sliders in it
        params, _ = data.get_params()
        self.param_sliders = {}
        for i, param in enumerate(['alpha', 'gamma', 'c_bb', 'c_rr', 'c_rb', 'c_br']):
            self.slider_ax = self.fig.add_axes([0.05, 0.85-0.05*i, 0.15, 0.03])#, axisbg='blue')
            self.param_sliders[param] = Slider(self.slider_ax, param, 0, 1, valinit=params[param], valfmt='%.2f')
            self.param_sliders[param].on_changed(change_param(param))

    def onpick(self, event):
        N = len(event.ind)
        if not N: return True

        # Collect all cells from patch
        patch_id = self.model.allVegetation[event.ind[0]].id
        cells = [cell for cell in self.model.allVegetation if cell.id == patch_id]
        RL_biom = [cell.biomass for cell in cells if cell.cell_type == 'RL']
        BR_biom = [cell.biomass for cell in cells if cell.cell_type == 'BR']
        comp_RL = -np.array([cell.grow_comp for cell in cells if cell.cell_type == 'RL'])
        comp_BR = -np.array([cell.grow_comp for cell in cells if cell.cell_type == 'BR'])
        conn = np.array([cell.grow_conn_loc for cell in cells]) - 1
        pos = np.array([cell.grow_pos for cell in cells]) - 1

        # Draw patch specific info
        fig = plt.figure(figsize=(5, 6))
        fig.suptitle(patch_id)
        gs = GridSpec(5, 2)
        ax1 = plt.subplot(gs[:-3, :])
        ax2 = plt.subplot(gs[-2:, :-1])
        ax3 = plt.subplot(gs[-2:, -1])

        ax1.hist(comp_RL, label='RL competition')
        ax1.hist(comp_BR, label='BR competition')
        ax1.hist(conn, label='Connectivity')
        ax1.hist(pos, label='Position')
        ax1.legend()
        ax1.set_xlabel("Relative intensity")
        ax1.set_title('Interaction mechanisms')

        ax2.hist(RL_biom)
        ax2.set_title("RL biomass")
        ax3.hist(BR_biom)
        ax3.set_title("BR biomass")

        fig.show()

    def update(self):
        # TODO: why difference in biomass between FIT and VIS?
        # removing old colorbar, updating grid and plotting new colorbar
        self.cax1.clear()
        # [t.remove() for t in self.gridtext]
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Setting all text right
        self.fig.suptitle("Plot %i simulation at day %i" % (self.nr, (self.model.time - start_date).days))
        self.ax1.set_title("Gridview")
        self.ax1.set_ylabel('Biomass (g)')
        self.ax2.set_xlabel("Day")
        self.ax2.set_ylabel("Average biomass (g)")
        self.ax3.set_title("Growth overview")
        self.ax3.set_ylabel("Relative intensity mechanisms")

        # Drawing plot
        self.BR_grid = np.zeros((self.model.width, self.model.height))
        for cell in self.model.vegetation["BR"]:
            if cell.biomass >= 0.021: # Making sure seedlings don't distort picture
                self.BR_grid[cell.pos] = cell.biomass
        self.BR_grid[self.BR_grid < 0.021] = 0.001
        im1 = self.ax1.imshow(self.BR_grid.T, cmap=plt.cm.Greens, norm=matplotlib.colors.LogNorm(vmin=0.001, vmax=0.6))
        self.fig.colorbar(im1, cax=self.cax1)
        self.cax1.tick_params(labelsize=7)

        self.RL_grid = np.zeros((self.model.width, self.model.height))
        RL_scatter = [(cell.pos, cell.biomass) for cell in self.model.vegetation["RL"]]
        for pos, biom in RL_scatter:
            self.ax1.scatter(*pos, facecolors='none', s=80*biom, edgecolors='r')


        # Generate markers for mouse click events
        vegetation = [(*cell.pos, cell.biomass) for cell in self.model.allVegetation]
        x, y, biomass = zip(*vegetation)
        self.ax1.plot(x, y, ',', alpha=0, picker=2)
        self.fig.canvas.mpl_connect('pick_event', self.onpick)

        self.gridtext = [
            self.ax1.text(*(patch.RL+patch.BR_original)[0].pos, '*', color='black', ha='center', va='center', fontsize=15)
            for patch in self.model.patches if patch.has_data
        ]

        # plotting biomass
        RL_mean, BR_mean, RL_measure, BR_measure, dates = self.get_biom_averages()
        day = len(self.model.data['biom_R'])
        x = range(day)
        self.ax2.plot(RL_mean, label="RL data", linestyle=':', color='r')
        self.ax2.plot(BR_mean, label="BR data", linestyle=':', color='g')
        self.ax2.scatter(dates, RL_measure, color='r')
        self.ax2.scatter(dates, BR_measure, color='g')
        self.ax2.errorbar(x, self.model.data['biom_R_measured'], color='r',
                      label="RL measured (%i)" % len([c for c in self.model.vegetation["RL"] if c.has_data])
                      )
        self.ax2.errorbar(x, np.array(self.model.data['biom_B_measured']), color='g',
                      label="BR measured (%i)" % (len([c for c in self.model.vegetation["BR"] if c.has_data])/9)
                      )
        # self.ax2.errorbar(x, self.model.data['biom_R'], yerr = self.model.data['biom_R_std'],
        #                   label = "RL all (%i)" % len(self.model.vegetation["RL"]))
        # self.ax2.errorbar(x, np.array(self.model.data['biom_B'])*9, yerr = np.array(self.model.data['biom_B_std'])*9,
        #                   label = "BR all (%i)" % (len(self.model.vegetation["BR"])/9))
        self.ax2.legend()

        # p = self.model.BR
        # K, m, r_bb, r_br = p["K"], p["m"], p["r_ib"], p["r_ir"]
        # def f(biom_R):
        #     a = K/2*(1-m-r_bb)
        #     b = a**2 - 4 * K * r_br * np.array(biom_R)
        #     return a + np.sqrt(b), a - np.sqrt(b)
        # c1, c2 = f(self.model.data['biom_R'])
        # self.ax2.plot(c1)
        # self.ax2.plot(c2)

        self.ax3.errorbar(x, -np.array(self.model.data['comp_RL']), yerr = self.model.data['comp_RL_std'],label = 'RL experienced competition')
        self.ax3.errorbar(x, -np.array(self.model.data['comp_BR']), yerr = self.model.data['comp_BR_std'],label = 'BR experienced competition')
        self.ax3.errorbar(x, np.array(self.model.data['conn']) - 1, yerr = self.model.data['conn_std'], label = 'Connectivity')
        self.ax3.errorbar(x, np.array(self.model.data['pos']) - 1, label = 'Position')#, yerr = self.model.data['pos_std'])
        self.ax3.legend()

        # # plotting effective biomass
        # self.ax3.imshow(self.model.RL_diff.T, cmap=plt.cm.Reds)
        # self.ax3.imshow(self.model.BR_diff.T, alpha=.5, cmap=plt.cm.Greens)

    def teardown(self):
        plt.ioff()

    def get_biom_averages(self):
        RL, BR = [], []
        for patch in self.model.patches:
            RL.append(patch.RL_data / patch.factor)
            BR.append(patch.BR_data / (patch.factor * 9))
        RL, BR = np.array(RL), np.array(BR)
        RL.flat[RL.flat == 0] = np.nan
        BR.flat[BR.flat == 0] = np.nan
        RL_mean = np.nanmean(RL, axis=0)
        BR_mean = np.nanmean(BR, axis=0)

        t = [(date - start_date).days for date in data.daterange()]
        dates = [(measurement - start_date).days for measurement in data.measurements]
        RL_spline = inter.InterpolatedUnivariateSpline(dates, RL_mean, k=2)
        BR_spline = inter.InterpolatedUnivariateSpline(dates, BR_mean, k=2)

        return RL_spline(t), BR_spline(t), RL_mean, BR_mean, dates


