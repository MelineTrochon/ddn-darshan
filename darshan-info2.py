import darshan
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import time
import sys
import math
import scipy.sparse as sp
import array
import seaborn as sns
import os

def usage():
    """Print the usage of the script"""
    print("Usage: python darshan-info.py <Darshan file or repository> [Options]")
    print("If no options, then all the options will be done")
    print("Options :")
    print("dxt_posix : Shows the I/O grouped by file or rank, as a function of time")
    print("\tit can be followed by : Rank, File, hostname or all, default is all")
    return


def max_count_length(dxt_posix, n):
    """Return the max count and length of the dxt_posix records"""
    max_count = 0
    max_length = 0
    for e in dxt_posix:
        max_count = max(max_count, e['write_count'], e['read_count'])
        max_length = n
    return max_count, max_length


def get_host(dxt_posix):
    """Return the number of hosts and a dictionnary to convert hostname to a number"""
    host = {}
    Y_size = 0
    for e in dxt_posix:
        if e['hostname'] not in host:
            host[e['hostname']] = Y_size
            Y_size += 1
    return Y_size, host


def dxt_posix_heatmap(i, argv):
    """Plot the heatmap of the dxt_posix records"""
    all_groups = {'rank', 'file', 'hostname'}
    groups = list()
    n=50
    path = argv[1]
    t = len(argv)
    output = "output/" + path.split('/')[-1].split('.')[0]
    output_bool = False
    norm_bool = False
    norm = 'linear'

    # Parse the arguments of argv 
    for t in range(i+1, len(argv)):
        
        if output_bool:
            output = argv[t]
            output_bool = False
            continue

        if norm_bool:
            norm = argv[t]
            norm_bool = False
            continue

        if argv[t] in all_groups:
            groups += {argv[t]}
            continue
        
        if argv[t].isdigit() :
            n = argv[t]
            continue
        
        if argv[t] == '-output':
            output_bool = True
            continue

        if argv[t] == '-norm':
            norm_bool = True
            continue
        
        break

    if len(groups) == 0:
        groups = all_groups
    
    print()
    print("="*100 + "\nStart reading dxt records sorted by {}\n".format(groups))

    # Read the dxt_posix records
    start_t = time.time()
    if os.path.isdir(path):
        dxt_posix, nprocs, start, end = merge_darshan_logs(path)
    
    elif path.split('.')[-1] == 'darshan' :
        dxt_posix, nprocs, start, end = darshan_file(path)
    
    else :
        print("The given path could not be read")
        exit(0)
    print("Read and convert all the data in %.3f sec\n"%(time.time() - start_t))

    # Plot the heatmap
    step = (end - start).total_seconds() / n
    for group in groups :
        plot_dxt_posix_heatmap(dxt_posix, group, nprocs, n, step, norm, output)
    return t


def darshan_file(file) :
    """Return the dxt_posix records of a darshan file"""
    if file.split('.')[-1] != 'darshan' :
        print("The file {} is not a darshan log and will be ignored".format(f))
        return

    report = darshan.DarshanReport(file, read_all=False)

    if 'DXT_POSIX' not in report.modules:
        print("The file {} does not contain dxt_posix records".format(f))
        return
    
    print("Reading the file {}".format(file), end='\t')
    start_t = time.time()
    report.read_all_dxt_records()
    print("read dxt_records in %.3f sec"%(time.time() - start_t), end='\t')

    start_t = time.time()
    dxt_posix = report.records['DXT_POSIX'].to_df()
    print("convert to df in %.3f sec"%(time.time() - start_t))
    
    return dxt_posix, report.metadata['job']['nprocs'], report.start_time, report.end_time


def merge_darshan_logs(path) :
    """Return the dxt_posix records of a darshan repository"""
    files = os.listdir(path)
    dxt_posix = None
    start = None
    end = None
    nprocs = 0

    for f in files:
        
        report_dxt_posix, report_nprocs, report_start, report_end = darshan_file(path + '/' + f)
        for line in report_dxt_posix:
            line['rank'] += nprocs
        
        nprocs += report_nprocs
        dxt_posix = dxt_posix + report_dxt_posix if dxt_posix is not None else report_dxt_posix

        start = report_start if start is None or report_start < start else start
        end = report_end if end is None or report_end > end else end

    return dxt_posix, nprocs, start, end


def plot_dxt_posix_heatmap(dxt_posix, group, nprocs, n, step, norm, output):
    """Compute sparses matrices then plot the heatmap of the dxt_posix records"""

    print("Plotting the heatmap for the group {}".format(group))
    start_t = time.time()
    max_count, max_length = max_count_length(dxt_posix, n)
    if group == 'rank' :
        Y_size = nprocs
    elif group == 'file':
        Y_size = len(dxt_posix)
    elif group == 'hostname' :
        Y_size, host = get_host(dxt_posix)
    else:
        print("The group {} is not valid".format(group))
        exit(0)

    x_read = np.zeros((len(dxt_posix), max_count, max_length), dtype=np.int32)
    y_read = np.zeros((len(dxt_posix), max_count, max_length), dtype=np.int32)
    x_write = np.zeros((len(dxt_posix), max_count, max_length), dtype=np.int32)
    y_write = np.zeros((len(dxt_posix), max_count, max_length), dtype=np.int32)
    v_read = np.zeros((len(dxt_posix), max_count, max_length), dtype=np.int64)
    v_write = np.zeros((len(dxt_posix), max_count, max_length), dtype=np.int64)
    v_read_bw = np.zeros((len(dxt_posix), max_count, max_length), dtype=np.int64)
    v_write_bw = np.zeros((len(dxt_posix), max_count, max_length), dtype=np.int64)


    def incrementation(line, it_posix, it_line, v, v_bw, x, y, i, step):
        idx = math.floor(line[3] / step)
        length = round((line[3] - line[2]) / 2) + 1
        for it in range(length) :
            x[it_posix, it_line, it] = i
            y[it_posix, it_line, it] = idx - it
            v[it_posix, it_line, it] = 1
            v_bw[it_posix, it_line, it] = int(line[1] / length)
        return length

    i = 0
    max_length = 0
    for it_posix, e in enumerate(dxt_posix):

        if group == 'rank' :
            i = e['rank']
        elif group == 'file' :
            i = it_posix
        elif group == 'hostname' :
            i = host[e['hostname']]
        else:
            print("The group {} is not valid".format(group))
            exit(0)
        
        if not e['read_segments'].empty:
            df = e['read_segments']
            for it_line, line in enumerate(df.values):
                length = incrementation(line, it_posix, it_line, v_read, v_read_bw, x_read, y_read, i, step)
                max_length = length if length > max_length else max_length

        if not e['write_segments'].empty:
            df = e['write_segments']
            for it_line, line in enumerate(df.values):
                length = incrementation(line, it_posix, it_line, v_write, v_write_bw, x_write, y_write, i, step)
                max_length = length if length > max_length else max_length
  
    shape = (Y_size, n + 1)
    print("max_length = {}".format(max_length))
    def sparse_matrix(x, y, v, v_bw, shape=shape):
        mat = sp.coo_matrix((v.flatten(), (x.flatten(), y.flatten())), shape=shape)
        mat_bw = sp.coo_matrix((v_bw.flatten(), (x.flatten(), y.flatten())), shape=shape)
        mat.sum_duplicates()
        mat_bw.sum_duplicates()
        mat.eliminate_zeros()
        mat_bw.eliminate_zeros()

        v_bw_count = mat_bw.data / mat.data
        mat_bw_count = sp.coo_matrix((v_bw_count, (mat.row, mat.col)), shape=shape)
        return mat, mat_bw, mat_bw_count

    read, read_bw, read_bw_count = sparse_matrix(x_read, y_read, v_read, v_read_bw)
    write, write_bw, write_bw_count = sparse_matrix(x_write, y_write, v_write, v_write_bw)

    print("done the sparses matrices in %.3f sec"%(time.time() - start_t))
    start_t = time.time()

    fig, axs = plt.subplots(3, 2, figsize=(7, 8))
    vmax = max(read.max(), write.max())
    vmax_bw = max(read_bw.max(), write_bw.max())
    vmax_bw_count = max(read_bw_count.max(), write_bw_count.max())
    extent = [0, step * n, 0, Y_size]

    def plot_heatmap(mat, ax, title, vmax=vmax, extent=extent, norm=norm):
        ax.set_title(title)
        ax.grid(True)
        return ax.imshow(mat.toarray(), interpolation='nearest', aspect='auto', origin='lower', cmap='Reds', vmax=vmax, extent=extent, norm=norm)

    plot_heatmap(read, axs[0,0], 'Read')
    plot_heatmap(write, axs[0,1], 'Write')
    plot_heatmap(read_bw, axs[1,0], 'Read BW', vmax=vmax_bw)
    plot_heatmap(write_bw, axs[1,1], 'Write BW', vmax=vmax_bw)
    plot_heatmap(read_bw_count, axs[2,0], 'Read BW count', vmax=vmax_bw_count)
    plot_heatmap(write_bw_count, axs[2,1], 'Write BW count', vmax=vmax_bw_count)

    fig.colorbar(axs[0,0].get_images()[0], ax=axs[0,0])
    fig.colorbar(axs[1,0].get_images()[0], ax=axs[1,0])
    fig.colorbar(axs[2,0].get_images()[0], ax=axs[2,0])
    axs[0,0].set_ylabel(group)
    axs[1,0].set_ylabel(group)
    axs[2,0].set_ylabel(group)
    axs[2,0].set_xlabel('Time (s)')
    axs[2,1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(output + '_' + group + '.png', dpi=300)
    print("done the plots in %.3f sec\n"%(time.time() - start_t))
    return


def main():
    argv = sys.argv
    if len(argv) < 2 or argv[1] == '-h' or argv[1] == '--help':
        usage()
        return 0
    
    if len(argv) == 2:
        start_t = time.time()
        dxt_posix_heatmap(2, argv)
        print("dxt_posix in %.3f sec\n"%(time.time() - start_t))

    i = 2
    while i < len(argv):

        if argv[i] == 'dxt_posix' :
            start_t = time.time()
            i = dxt_posix_heatmap(i, argv)
            print("dxt_posix in %.3f sec\n"%(time.time() - start_t))
        
        else : 
            print("The option " + str(argv[i]) + " is not recognized")

        i += 1

main()