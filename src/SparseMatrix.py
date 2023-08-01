import scipy.sparse as sp

"""
This class is a sub-class that creates Sparse Matrices from 3 array in a coo_format then plotted it as heatmaps.
"""


class SparseMatrix:
    def __init__(self, info, v, x, y, shape, dtype="float64"):
        self.mat = sp.coo_matrix((v, (x, y)), shape=shape, dtype=dtype)
        self.mat.sum_duplicates()
        self.vmax = self.mat.data.max() if len(self.mat.data) != 0 else 0
        self.extent = [0, info.duration, 0, shape[0]]
        self.norm = info.norm

    def plot_heatmap(self, ax, title):
        ax.imshow(
            self.mat.toarray(),
            aspect="auto",
            cmap="Reds",
            interpolation="nearest",
            origin="lower",
            extent=self.extent,
            norm=self.norm,
            vmax=self.vmax,
        )
        ax.set_title(title)
        ax.grid(True)
