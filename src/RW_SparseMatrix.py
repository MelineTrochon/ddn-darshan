import array
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd

from SparseMatrix import SparseMatrix

"""
This class is used to create a heatmap that shows : 
the number of I/O operations, 
the bandwith, 
the average bandwitdh per operation, 
and the average metadata per operation

It can be grouped by rank, file or hostname

The options are :
    - op : the type of operation (read, write, or both)
    - group : the group by which the data will be grouped (rank, file, or hostname)
    - norm : the normalization of the heatmap (see matplotlib.pyplot.imshow documentation)
    - nbins : the number of bins for the heatmap (i.e the discretisation of the time axis)
    - output : the name of the output repository
"""


class RW_SparseMatrix:
    def __init__(self, info):
        self.shape = (self.get_Y_size(info), info.nbins + 1)
        self.x = array.array('I')
        self.y = array.array('I')
        self.v = array.array('L')
        self.v_bw = array.array('d')
        self.v_meta = array.array('d')
        self.get_data(info)
    
    def get_Y_size(self, info):
        if info.group == 'rank':
            return info.nprocs
        elif info.group == 'file':
            return info.len_dxt_posix
        elif info.group == 'hostname':
            return len(info.hostnames)
        print("Wrong group name : {}".format(info.group))
        exit(1)

    def get_x(self, info, e, temps_nprocs):
        if info.group == 'rank':
            return temps_nprocs + e['rank']
        elif info.group == 'file':
            return info.file_ids.index(e['id'])
        elif info.group == 'hostname':
            return info.hostnames.index(e['hostname'])
        print("Wrong group name : {}".format(info.group))
        exit(1)

    def get_data(self, info):
        temps_nprocs = 0
        op_segment = info.op + '_segments'
        for f in info.files:
            posix = pd.merge(f.posix['counters'], f.posix['fcounters'], on=['id', 'rank'])
            meta_time = posix[((posix['POSIX_READS'] != 0) | (posix['POSIX_WRITES'] != 0))]
            meta_time = meta_time.set_index(['id', 'rank'])['POSIX_F_META_TIME'].to_dict()

            for e in f.dxt_posix:
                x = self.get_x(info, e, temps_nprocs)
                meta = meta_time.get((e['id'], e['rank']), 0) / (e['read_count'] + e['write_count'])
                if meta == 0:
                    print("meta time has not been found for ({}, {})".format(e['id'], e['rank']))
                if not e[op_segment].empty:
                    df = e[op_segment]
                    for line in df.values:
                        idx_beg = int(line[2] // info.step)
                        duration = line[3] - line[2]
                        length = int(duration // info.step + 1)
                        if idx_beg + length > info.nbins:
                            print(line[2], line[3], info.step, info.duration)
                            print("Warning : the segment is out of the range of the histogram")
                            length = info.nbins - idx_beg

                        for it in range(length):
                            self.x.append(x)
                            self.y.append(idx_beg + it)
                            self.v.append(1)
                            self.v_bw.append(line[1] / length)
                            self.v_meta.append(meta / length)
                            if meta / length > 1 :
                                print("meta / length : {}".format(meta / length))

            temps_nprocs += f.nprocs

        self.v_bw_count = [self.v_bw[i] / self.v[i] for i in range(len(self.v))]

        self.mat = SparseMatrix(info, self.v, self.x, self.y, shape=self.shape, dtype='int64')
        self.mat_bw = SparseMatrix(info, self.v_bw, self.x, self.y, shape=self.shape)
        self.mat_bw_count = SparseMatrix(info, self.v_bw_count, self.x, self.y, shape=self.shape)
        self.mat_meta = SparseMatrix(info, self.v_meta, self.x, self.y, shape=self.shape)
        return

    def to_heatmap(self, info):
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle('Heatmap of the {} I/O grouped by {}'.format(info.op, info.group))

        self.mat.plot_heatmap(axs[0, 0], str(info.op) + ' count')
        self.mat_bw.plot_heatmap(axs[0, 1], str(info.op) + ' bandwidth')
        self.mat_bw_count.plot_heatmap(axs[1, 0], str(info.op) + ' bandwidth per I/O')
        self.mat_meta.plot_heatmap(axs[1, 1], str(info.op) + ' metadata')

        fig.colorbar(axs[0, 0].images[0], ax=axs[0, 0])
        fig.colorbar(axs[0, 1].images[0], ax=axs[0, 1])
        fig.colorbar(axs[1, 0].images[0], ax=axs[1, 0])
        fig.colorbar(axs[1, 1].images[0], ax=axs[1, 1])

        axs[0, 0].set_ylabel('{}'.format(info.group))
        axs[1, 0].set_ylabel('{}'.format(info.group))
        axs[1, 0].set_xlabel('Time ({} s)'.format(info.step))
        axs[1, 1].set_xlabel('Time ({} s)'.format(info.step))

        fig.tight_layout()
        fig.savefig('{}_{}_{}.png'.format(info.output, info.op, info.group))
        return
