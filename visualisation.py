import matplotlib
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

class Visualize(object):
    def __init__(self, model, nr):
        self.model = model
        self.nr = nr

    def initiate(self):
        # Starting interactive matplotlib plot
        plt.ion()
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.suptitle("Plot %i simulation" % self.nr)

        # Dividing matplotlib figure into three sections
        gs = GridSpec(3, 3)
        self.ax1 = plt.subplot(gs[:, :-1])
        self.ax1.set_title("Gridview")
        # self.ax1.yaxis.set_label_position("right")
        self.ax1.set_ylabel('Biomass (g)')
        self.ax2 = plt.subplot(gs[-1, -1])
        self.ax2.set_title("Total biomass")
        self.ax2.set_xlabel("Timestep")
        self.ax2.set_ylabel("Biomass")
        self.ax3 = plt.subplot(gs[:-1, -1])
        self.ax3.set_title("Effective biomass view")

        # Make grid, colorbar and text on grid
        self.cax = make_axes_locatable(self.ax1).append_axes("right", size="5%", pad="2%")
        self.cax.set_xlabel('hoi')
        self.grid = np.empty((self.model.width, self.model.height))
        for cell in self.model.grid.flatten():
            self.grid[cell.pos] = cell.biomass

        # self.gridtext = [
        #     self.ax1.text(i, j, int(self.model.grid[i, j].upslope), color='white', ha='center', va='center')
        #     for i, j in zip(*self.grid.nonzero())
        # ]

        self.update()

    def update(self):
        for cell in self.model.grid.flatten():
            self.grid[cell.pos] = cell.biomass

        # removing old colorbar, updating grid and plotting new colorbar
        self.cax.clear()
        # [t.remove() for t in self.gridtext]
        self.ax2.clear()

        im = self.ax1.imshow(self.grid.T, cmap=plt.cm.Greens, norm=matplotlib.colors.LogNorm())
        self.fig.colorbar(im, cax=self.cax)
        # self.gridtext = [
        #     self.ax1.text(i, j, int(self.model.grid[i, j].upslope), color='white', ha='center', va='center')
        #     for i, j in zip(*self.grid.nonzero())
        # ]

        # plotting biomass
        x = range(len(self.model.data['biomass']))
        self.ax2.errorbar(x, self.model.data['biomass'], yerr = self.model.data['biomass_std'])

        # self.ax2.errorbar(x, self.model.data['comp'], yerr = self.model.data['comp_std'],label = 'Competition')
        # self.ax2.errorbar(x, self.model.data['conn'], yerr = self.model.data['conn_std'], label = 'Connectivity')
        # self.ax2.errorbar(x, self.model.data['pos'], yerr = self.model.data['pos_std'], label = 'Position')
        # self.ax2.legend()

        # plotting effective biomass
        self.ax3.imshow(self.model.RL_diff.T, cmap=plt.cm.Reds)
        self.ax3.imshow(self.model.BR_diff.T, alpha=.5, cmap=plt.cm.Greens)

    def teardown(self):
        plt.ioff()