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

def dxt_posix_sorted_RW (report, sep='Rank', n=50) :
    if not 'DXT_POSIX' in report.modules :
        print("The file does not contain dxt_posix records")
        exit(1)

    if not 'DXT_POSIX' in report.records.keys() :
        start = time.time()
        report.read_all_dxt_records()
        print("Read all dxt records in {} second(s)".format(int(time.time() - start)))
        report.info()
    
    start = time.time()
    dxt_posix = report.records['DXT_POSIX'].to_df()
    print("Convert dxt records to dataframe in {} second(s)".format(int(time.time() - start)))
    duration = report.end_time - report.start_time
    step = int(duration.total_seconds()) / n
    if sep == 'Rank' :
        Y_size = report.metadata['job']['nprocs']
    if sep == 'File' :
        Y_size = len(dxt_posix)

    shape = (Y_size, n + 1)
    x_read = array.array('I')
    y_read = array.array('I')
    x_write = array.array('I')
    y_write = array.array('I')
    v_read = array.array('I')
    v_write = array.array('I')
    v_read_bw = array.array('L')
    v_write_bw = array.array('L')

    start = time.time()
    idx_prec_r = -1
    i_prec_r = -1
    idx_prec_w = -1
    i_prec_w = -1
    repet = 0

    for it, e in enumerate(dxt_posix):
        if sep == 'Rank' :
            i = e['rank']
        if sep == 'File' :
            i = it
        if not e['read_segments'].empty:
            df = e['read_segments']
            for line in df.values:
                time_t = (line[3] + line[2]) / 2
                idx = math.floor(time_t / step)
                if i_prec_r == i and idx_prec_r == idx :
                    v_read[-1] += 1
                    v_read_bw[-1] += int(line[1])
                else :
                    x_read.append(i)
                    y_read.append(idx)
                    v_read.append(1)
                    v_read_bw.append(int(line[1]))
                i_prec_r = i
                idx_prec_r = idx
        if not e['write_segments'].empty:
            df = e['write_segments']
            for line in df.values:
                time_t = (line[3] + line[2]) / 2
                idx = math.floor(time_t / step)
                if i_prec_w == i and idx_prec_w == idx :
                    v_write[-1] += 1
                    v_write_bw[-1] += int(line[1])
                else : 
                    x_write.append(i)
                    y_write.append(idx)
                    v_write.append(1)
                    v_write_bw.append(int(line[1]))
                i_prec_w = i
                idx_prec_w = idx
    
    read = sp.coo_matrix((v_read, (x_read, y_read)), shape=shape)
    write = sp.coo_matrix((v_write, (x_write, y_write)), shape=shape)
    read_bw = sp.coo_matrix((v_read_bw, (x_read, y_read)), shape=shape)
    write_bw = sp.coo_matrix((v_write_bw, (x_write, y_write)), shape=shape)

    v_read = np.array(v_read)
    v_read_bw = np.array(v_read_bw)
    read_bw_count  = sp.coo_matrix((v_read_bw / v_read, (x_read, y_read)), shape=shape)
    v_write = np.array(v_write)
    v_write_bw = np.array(v_write_bw)
    write_bw_count  = sp.coo_matrix((v_write_bw / v_write, (x_write, y_write)), shape=shape)
    
    read.sum_duplicates()
    write.sum_duplicates()
    read_bw.sum_duplicates()
    write_bw.sum_duplicates()
    read_bw_count.sum_duplicates()
    write_bw_count.sum_duplicates()

    print("Create sparse matrices in {} second(s)".format(int(time.time() - start)))

    start = time.time()
    fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(3, 2, sharex=True, sharey='row', figsize=(7, 8))
    vmax = max(read.max(), write.max())
    vmax_bw = max(read_bw.max(), write_bw.max())
    vmax_bw_count = max(read_bw_count.max(), write_bw_count.max())
    extent = [0, int(duration.total_seconds()), 0, Y_size]
    output = (report.filename.split('/')[-1]).split('.')[-2]

    ax0.set_title('Reads')
    ax0.set_ylabel(sep)
    ax0.grid(True)
    im_read = ax0.imshow(read.toarray(), cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax)
    fig.colorbar(im_read, ax=ax0)

    ax1.set_title('Writes')
    ax1.grid(True)
    im_write = ax1.imshow(write.toarray(), cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax)

    ax2.set_title('Reads Bandwidth')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(sep)
    ax2.grid(True)
    im_read_bw = ax2.imshow(read_bw.toarray(), cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax_bw)
    fig.colorbar(im_read_bw, ax=ax2)

    ax3.set_title('Writes Bandwidth')
    ax3.set_xlabel('Time (s)')
    ax3.grid(True)
    im_write_bw = ax3.imshow(write_bw.toarray(), cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax_bw)

    ax4.set_title('Reads Bandwidth over Counts')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel(sep)
    ax4.grid(True)
    im_read_bw_count = ax4.imshow(read_bw_count.toarray(), cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax_bw_count)
    fig.colorbar(im_read_bw_count, ax=ax4)

    ax5.set_title('Writes Bandwidth over Counts')
    ax5.set_xlabel('Time (s)')
    ax5.grid(True)
    im_write_bw_count = ax5.imshow(write_bw_count.toarray(), cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax_bw_count)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.suptitle('I/O separated by {} as a function of time'.format(sep))
    plt.savefig('{}_dxt_posix_{}.png'.format(output, sep))
    print("Plot in {} second(s)".format(int(time.time() - start)))

def main ():
    argv = sys.argv
    if len(argv) < 2:
        print("Usage: python script.py <darshan log file>, see -h or --help for more information")
        exit(1)
    
    filename = argv[1]
    report = darshan.DarshanReport(filename, read_all=False)
    report.info()

    for i in range(2, len(argv)) :

        if argv[i] == '-h' or argv[i] == '--help' :
            print("Usage: python script.py <darshan log file> [options]")
            print("Options:")
            print("dxt_posix : Shows the I/O separated by file or rank, as a function of time")
            print("followed by : rank, file or all, nothing is equivalent to all)")
            exit(0)
        
        if argv[i] == 'dxt_posix' :
            if i + 1 >= len(argv) or not(argv[i+1] == 'File' or argv[i+1] == 'Rank' or argv[i+1] == 'all') :   
                dxt_posix_sorted_RW(report, 'Rank')
                dxt_posix_sorted_RW(report, 'File')
            else :
                dxt_posix_sorted_RW(report, argv[i+1])
                i += 1

        if argv[i] == 'dxt_posix_bw' :
            if i + 1 >= len(argv) or not(argv[i+1] == 'File' or argv[i+1] == 'Rank' or argv[i+1] == 'all') :   
                dxt_posix_bw(report, 'Rank')
                dxt_posix_bw(report, 'File')
            else :
                dxt_posix_bw(report, argv[i+1])
                i += 1

main()