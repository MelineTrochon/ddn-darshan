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

def merge (path) :
    files = os.listdir(path)
    nprocs = len(files)
    dxt_posix = None
    start = None
    end = None

    for i, f in enumerate(files) :
        if f.split('.')[-1] != 'darshan' :
            print("The file {} is not a darshan log thus is ignored".format(f))
            continue
        report = darshan.DarshanReport(path + '/' + f, read_all=False)

        if not 'DXT_POSIX' in report.modules :
            print("The file {} does not contain dxt_posix records, thus is ignored".format(f))
            continue
        report.read_all_dxt_records()
        report_dxt_posix = report.records['DXT_POSIX'].to_df()
        for line in report_dxt_posix :
            line['rank'] = i
        dxt_posix = dxt_posix + report_dxt_posix if dxt_posix is not None else report_dxt_posix

        start = report.start_time if not(start) or report.start_time < start else start
        end = report.end_time if not(end) or report.end_time > end else end

    duration = end - start
    return dxt_posix, nprocs, duration, path


def dxt_posix_sorted_RW (path, seps='Rank', n=50) :

    print()
    print("=="*20)
    print("Start reading dxt records sorted by {}".format(seps))

    if path.split('.')[-1] == 'darshan' :
        report = darshan.DarshanReport(path)

        if not 'DXT_POSIX' in report.modules :
            print("The file does not contain dxt_posix records")
            exit(1)

        if not 'DXT_POSIX' in report.records.keys() :
            start = time.time()
            report.read_all_dxt_records()
            print("Read all dxt records in %.3f sec"%(time.time() - start))
    
        start = time.time()
        dxt_posix = report.records['DXT_POSIX'].to_df()
        print("Convert dxt records to dataframe in %.3f sec"%(time.time() - start))

        nprocs = report.metadata['job']['nprocs']
        duration = report.end_time - report.start_time
        output = report.filename.split('/')[-1].split('.')[0] 

    else :
        start = time.time()
        dxt_posix, nprocs, duration, output = merge(path)
        print("Merge dxt records in %.3f sec"%(time.time() - start))

    step = int(duration.total_seconds()) / n
    for sep in seps :
        if sep == 'Rank' :
            Y_size = nprocs
        if sep == 'File' :
            Y_size = len(dxt_posix)

        start = time.time()
        read, write, read_bw, write_bw, read_bw_count, write_bw_count = _dxt_posix_sorted_RW (sep, n, dxt_posix, step, Y_size)
        print("Create sparse matrices in %.3f sec"%(time.time() - start))

        start = time.time()
        plot(read, write, read_bw, write_bw, read_bw_count, write_bw_count, duration, Y_size, output, sep)
        print("Plot in %.3f sec"%(time.time() - start))
    return

def _dxt_posix_sorted_RW (sep, n, dxt_posix, step, Y_size) :

    shape = (Y_size, n + 1)
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

    def incrementation(line, step, i, v, v_bw, x, y, i_prec, idx_prec) :
        idx = math.floor(line[3] / step)
        length = round((line[3] - line[2]) / 2) + 1
        loop = True
        if idx == idx_prec and i == i_prec :
            loop = False
            for it in range(length) :
                if x[-1 - it] == i and y[-1 - it] == idx :
                    v[-1 - it] += 1
                    v_bw[-1 - it] += int(line[1] / length)
                else :
                    length = length - it
                    loop = True
                    break

        if loop :
            for it in range(length) :
                x.append(i)
                y.append(idx - it)
                v.append(1)
                v_bw.append(int(line[1] / length))
            i_prec = i
            idx_prec = idx

    for it, e in enumerate(dxt_posix):
        if sep == 'Rank' :
            i = e['rank']
        if sep == 'File' :
            i = it
        if not e['read_segments'].empty:
            df = e['read_segments']
            for line in df.values:
                incrementation(line, step, i, v_read, v_read_bw, x_read, y_read, i_prec_r, idx_prec_r)

        if not e['write_segments'].empty:
            df = e['write_segments']
            for line in df.values:
                incrementation(line, step, i, v_write, v_write_bw, x_write, y_write, i_prec_w, idx_prec_w)
    
    read = sp.coo_matrix((v_read, (x_read, y_read)), shape=shape)
    write = sp.coo_matrix((v_write, (x_write, y_write)), shape=shape)
    read_bw = sp.coo_matrix((v_read_bw, (x_read, y_read)), shape=shape)
    write_bw = sp.coo_matrix((v_write_bw, (x_write, y_write)), shape=shape)
    
    read.sum_duplicates()
    write.sum_duplicates()
    read_bw.sum_duplicates()
    write_bw.sum_duplicates()

    v_read_bw_count = read_bw.data / read.data
    v_write_bw_count = write_bw.data / write.data
    read_bw_count  = sp.coo_matrix((v_read_bw_count, (read.row, read.col)), shape=shape)
    write_bw_count  = sp.coo_matrix((v_write_bw_count, (write.row, write.col)), shape=shape)

    return read, write, read_bw, write_bw, read_bw_count, write_bw_count


def plot(read, write, read_bw, write_bw, read_bw_count, write_bw_count, duration, Y_size, output, sep) :
    fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(3, 2, sharex=True, sharey='row', figsize=(7, 8))
    vmax = max(read.max(), write.max())
    vmax_bw = max(read_bw.max(), write_bw.max())
    vmax_bw_count = max(read_bw_count.max(), write_bw_count.max())
    extent = [0, int(duration.total_seconds()), 0, Y_size]
    # output = (report.filename.split('/')[-1]).split('.')[-2]

    ax0.set_title('Reads')
    ax0.set_ylabel(sep)
    ax0.grid(True)
    im_read = ax0.imshow(read.toarray(), cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax, interpolation='nearest', norm='linear')
    # im_read = sns.jointplot(x=read.col, y=read.row, kind="hist", space=0.05, ax=ax0)
    fig.colorbar(im_read, ax=ax0)

    ax1.set_title('Writes')
    ax1.grid(True)
    im_write = ax1.imshow(write.toarray(), cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax, interpolation='nearest', norm='linear')

    ax2.set_title('Reads Bandwidth')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(sep)
    ax2.grid(True)
    im_read_bw = ax2.imshow(read_bw.toarray(), cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax_bw, interpolation='nearest', norm='linear')
    fig.colorbar(im_read_bw, ax=ax2)

    ax3.set_title('Writes Bandwidth')
    ax3.set_xlabel('Time (s)')
    ax3.grid(True)
    im_write_bw = ax3.imshow(write_bw.toarray(), cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax_bw, interpolation='nearest', norm='linear')

    ax4.set_title('Reads Bandwidth over Counts')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel(sep)
    ax4.grid(True)
    im_read_bw_count = ax4.imshow(read_bw_count.toarray(), cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax_bw_count, interpolation='nearest', norm='linear')
    fig.colorbar(im_read_bw_count, ax=ax4)

    ax5.set_title('Writes Bandwidth over Counts')
    ax5.set_xlabel('Time (s)')
    ax5.grid(True)
    im_write_bw_count = ax5.imshow(write_bw_count.toarray(), cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax_bw_count, interpolation='nearest', norm='linear')

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.suptitle('I/O sorted by {} as a function of time'.format(sep))
    # plt.show()
    plt.savefig('{}_dxt_posix_{}.png'.format(output, sep))
    return


def main ():
    argv = sys.argv
    if len(argv) < 2:
        print("Usage: python script.py <darshan log file>, see -h or --help for more information")
        exit(1)
    
    if argv[1] == '-h' or argv[1] == '--help' :
        print("Usage: python script.py <darshan log file> [Options]")
        print()
        print("Options:")
        print("dxt_posix : Shows the I/O sorted by file or rank, as a function of time")
        print("\tfollowed by : Rank, File or All, nothing is equivalent to all")
        exit(0)
    
    path = argv[1]

    if path.split('.')[-1] == 'darshan' :
        report = darshan.DarshanReport(path, read_all=False)
        report.info()

    if len(argv) == 2 :
        dxt_posix_sorted_RW(path, {'Rank', 'File'})

    for i in range(2, len(argv)) :

        if argv[i] == 'dxt_posix' :

            n=50
            start_dxt_posix = time.time()
            if i + 1 >= len(argv) or argv[i+1] == 'All' :
                seps = {'Rank', 'File'}

            elif argv[i+1] == 'Rank' or argv[i+1] == 'File' :
                seps = {argv[i+1]}
                i += 1
            
            if argv[i+2] == '-n' :
                n = int(argv[i+3])
                i += 2
            
            print(n)
            dxt_posix_sorted_RW(path, seps, n)

            print("dxt_posix in %.3f sec"%(time.time() - start_dxt_posix))

main()