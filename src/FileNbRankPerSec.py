import array
import numpy as np
import matplotlib.pyplot as plt

from SparseMatrix import SparseMatrix

"""
This class is used to create a heatmap that shows the number of ranks that access a file each second.
"""


class FileNbRankPerSec:
    def __init__(self, info):
        self.shape = (info.len_dxt_posix, info.nbins + 1)
        self.x = [array.array("I") for i in range(info.nprocs)]
        self.y = [array.array("I") for i in range(info.nprocs)]
        self.value = [array.array("I") for i in range(info.nprocs)]
        self.get_data(info)

    def get_data(self, info):
        temps_nprocs = 0
        op_segment = info.op + "_segments"
        for f in info.files:
            for e in f.dxt_posix:
                x = info.file_ids.index(e["id"])
                if not e[op_segment].empty:
                    df = e[op_segment]
                    for line in df.values:
                        idx_beg = int(line[2] // info.step)
                        duration = line[3] - line[2]
                        length = int(duration // info.step + 1)
                        if idx_beg + length > info.nbins:
                            print(line[2], line[3], info.step, info.duration)
                            print(
                                "Warning : the segment is out of the range of the histogram"
                            )
                            length = info.nbins - idx_beg

                        for it in range(length):
                            self.x[temps_nprocs + e["rank"]].append(x)
                            self.y[temps_nprocs + e["rank"]].append(idx_beg + it)
                            self.value[temps_nprocs + e["rank"]].append(1)
            temps_nprocs += f.nprocs

        self.rank_mat = [
            SparseMatrix(
                info,
                self.value[i],
                self.x[i],
                self.y[i],
                shape=self.shape,
                dtype="int32",
            )
            for i in range(info.nprocs)
        ]

        for i in range(info.nprocs):
            self.rank_mat[i].mat.data = np.ones(
                len(self.rank_mat[i].mat.data), dtype="int32"
            )
        self.mat = self.rank_mat[0].mat.tocsr()
        for i in range(1, info.nprocs):
            self.mat += self.rank_mat[i].mat.tocsr()

        return

    def to_heatmap(self, info):
        fig, axs = plt.subplots(1, 1)
        fig.suptitle(
            "Heatmap of the number of rank {}ing on the same file".format(info.op)
        )
        extent = [0, info.duration, 0, info.len_dxt_posix]
        vmax = max(self.mat.data) if len(self.mat.data) > 0 else 0
        norm = info.norm

        def plot_heatmap(self, ax, title):
            ax.imshow(
                self.mat.toarray(),
                aspect="auto",
                cmap="Reds",
                interpolation="nearest",
                origin="lower",
                extent=extent,
                norm=norm,
                vmax=vmax,
            )
            ax.set_title(title)
            ax.grid(True)

        plot_heatmap(self, axs, str(info.op))
        fig.colorbar(axs.images[0], ax=axs)
        fig.tight_layout()
        fig.savefig("{}_{}_nrank.png".format(info.output, info.op))
