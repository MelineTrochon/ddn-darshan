import array
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from SparseMatrix import SparseMatrix


class AggregateInfo:
    def __init__(self, info):
        self.size = info.nbins
        # self.v = array.array('L')
        # self.v_bw = array.array('d')
        # self.v_meta = array.array('d')
        self.v = np.zeros(self.size, dtype=np.uint64)
        self.v_bw = np.zeros(self.size, dtype=np.float64)
        self.v_meta = np.zeros(self.size, dtype=np.float64)
        self.get_data(info)

    def get_data(self, info):
        op_segment = info.op + "_segments"
        temps_nprocs = 0

        for f in info.files:
            posix = pd.merge(
                f.posix["counters"], f.posix["fcounters"], on=["id", "rank"]
            )
            meta_time = posix[
                ((posix["POSIX_READS"] != 0) | (posix["POSIX_WRITES"] != 0))
            ]
            meta_time = meta_time.set_index(["id", "rank"])[
                "POSIX_F_META_TIME"
            ].to_dict()

            for e in f.dxt_posix:
                meta = meta_time.get((e["id"], e["rank"]), 0) / (
                    e["read_count"] + e["write_count"]
                )
                if meta == 0:
                    print(
                        "meta time has not been found for ({}, {})".format(
                            e["id"], e["rank"]
                        )
                    )
                if not e[op_segment].empty:
                    df = e[op_segment]
                    for line in df.values:
                        idx_beg = int(line[2] // info.step)
                        # duration = line[3] - line[2]
                        idx_end = int(line[3] // info.step)
                        # length = int(duration // info.step + 1)
                        length = idx_end - idx_beg + 1
                        if idx_beg + length > info.nbins:
                            print(line[2], line[3], info.step, info.duration)
                            print(
                                "Warning : the segment is out of the range of the histogram"
                            )
                            length = info.nbins - idx_beg

                        self.v[idx_beg : idx_beg + length] += 1
                        self.v_bw[idx_beg : idx_beg + length] += line[1] / length
                        self.v_meta[idx_beg : idx_beg + length] += meta / length

            temps_nprocs += f.nprocs

        self.v_bw_count = np.zeros(self.size, dtype=np.float64)
        for i in range(self.size):
            if self.v[i] != 0:
                self.v_bw_count[i] = self.v_bw[i] / self.v[i]

        return

    def to_plot(self, info):
        fig, ax = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle("Plot of the {} I/O".format(info.op))

        ax[0, 0].plot(self.v, label="Number of {}s".format(info.op))
        ax[0, 0].set_title("Number of {}s".format(info.op))
        # ax[0, 0].set_xlabel('Time (s)')
        ax[0, 0].set_ylabel("Number of {}s".format(info.op))

        ax[0, 1].plot(self.v_bw, label="Bandwidth of {}s".format(info.op))
        ax[0, 1].set_title("Bandwidth of {}s".format(info.op))
        # ax[0, 1].set_xlabel('Time (s)')
        ax[0, 1].set_ylabel("Bandwidth of {}s".format(info.op))

        ax[1, 0].plot(self.v_bw_count, label="Bandwidth per {}s".format(info.op))
        ax[1, 0].set_title("Bandwidth per {}s".format(info.op))
        ax[1, 0].set_xlabel("Time (s)")
        ax[1, 0].set_ylabel("Bandwidth per {}s".format(info.op))

        ax[1, 1].plot(self.v_meta, label="Metadata time per {}s".format(info.op))
        ax[1, 1].set_title("Metadata time per {}s".format(info.op))
        ax[1, 1].set_xlabel("Time (s)")
        ax[1, 1].set_ylabel("Metadata time per {}s".format(info.op))

        fig.tight_layout()
        fig.savefig("{}_{}_plot.png".format(info.output, info.op))
        return
