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
    print("Usage: python darshan-info.py <Darshan file or repository> [Options]")
    print("If no options, then all the options will be done")
    print("Options :")
    print("dxt_posix : Shows the I/O grouped by file or rank, as a function of time")
    print("\tit can be followed by : Rank, File, hostname or all, default is all")
    return


def dxt_posix_heatmap(i, argv):
    all_groups = {'rank', 'file', 'hostname'}
    groups = {}
    n=50
    path = argv[1]
    t = len(argv)
    output = "output/" + path.split('/')[-1].split('.')[0]
    output_bool = False
    norm_bool = False
    norm = 'linear'


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

    start_t = time.time()
    if os.path.isdir(path):
        dxt_posix, nprocs, start, end = merge_darshan_logs(path)
    
    elif path.split('.')[-1] == 'darshan' :
        dxt_posix, nprocs, start, end = darshan_file(path)
    
    else :
        print("The given path could not be read")
        exit(0)
    print("Read and convert all the data in %.3f sec\n"%(time.time() - start_t))

    step = (end - start).total_seconds() / n
    for group in groups :
        plot_dxt_posix_heatmap(dxt_posix, group, nprocs, n, step, norm, output)
    return t


def darshan_file(file) :

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

    print("Plotting the heatmap for the group {}".format(group))
    start_t = time.time()
    x_read = array.array('I')
    y_read = array.array('I')
    x_write = array.array('I')
    y_write = array.array('I')
    v_read = array.array('I')
    v_write = array.array('I')
    v_read_bw = array.array('L')
    v_write_bw = array.array('L')

    idx_prec_r = -1
    i_prec_r = -1
    idx_prec_w = -1
    i_prec_w = -1

    if group == 'hostname':
        host = {}

    def incrementation(line, step, i, v, v_bw, x, y, i_prec, idx_prec):
        incr = 0
        idx = math.floor(line[3] / step)
        length = round((line[3] - line[2]) / 2) + 1
        for it in range(length) :
            x.append(i)
            y.append(idx - it)
            v.append(1)
            v_bw.append(int(line[1] / length))
        return

    i = 0
    for it, e in enumerate(dxt_posix):
        if group == 'rank' :
            i = e['rank']
        elif group == 'file' :
            i = it
        elif group == 'hostname' :
            if e['hostname'] not in host :
                host[e['hostname']] = len(host)
            else :
                i = host[e['hostname']]
        else:
            print("The group {} is not valid".format(group))
            exit(0)
        
        if not e['read_segments'].empty:
            df = e['read_segments']
            for line in df.values:
                incrementation(line, step, i, v_read, v_read_bw, x_read, y_read, i_prec_r, idx_prec_r)

        if not e['write_segments'].empty:
            df = e['write_segments']
            for line in df.values:
                incrementation(line, step, i, v_write, v_write_bw, x_write, y_write, i_prec_w, idx_prec_w)

    if group == 'rank' :
        Y_size = nprocs
    elif group == 'file' :
        Y_size = len(dxt_posix)
    elif group == 'hostname' :
        Y_size = len(host)
    else:
        print("The group {} is not valid".format(group))
        exit(0)
    
    shape = (Y_size, n + 1)

    def sparse_matrix(x, y, v, v_bw, shape=shape):
        mat = sp.coo_matrix((v, (x, y)), shape=shape)
        mat_bw = sp.coo_matrix((v_bw, (x, y)), shape=shape)
        mat.sum_duplicates()
        mat_bw.sum_duplicates()

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