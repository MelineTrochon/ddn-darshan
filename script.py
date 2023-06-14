import darshan
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import time
import sys
import math

def dxt_posix_sorted_RW (report, sep='Rank') :
    if not 'DXT_POSIX' in report.modules :
        print("The file does not contain dxt_posix records")
        exit(1)

    if not 'DXT_POSIX' in report.records.keys() :
        report.read_all_dxt_records()
        report.info()
    
    dxt_posix = report.records['DXT_POSIX'].to_df()
    duration = report.end_time - report.start_time
    n = 50
    step = int(duration.total_seconds()) / n
    if sep == 'Rank' :
        Y_size = report.metadata['job']['nprocs']
    if sep == 'File' :
        Y_size = len(dxt_posix)
    # X = np.linspace(0, int(duration.total_seconds()), n)
    read = np.zeros((Y_size, n + 1))
    read_bw = np.zeros((Y_size, n + 1))
    write = np.zeros((Y_size, n + 1))
    write_bw = np.zeros((Y_size, n + 1))

    for it, e in enumerate(dxt_posix):
        if sep == 'Rank' :
            i = e['rank']
        if sep == 'File' :
            i = it
        if not e['read_segments'].empty:
            df = e['read_segments']
            for line in df.values:
                time = (line[3] + line[2]) / 2
                idx = math.floor(time / step)
                read[i, idx] += 1
                read_bw[i, idx] += line[1]
        if not e['write_segments'].empty:
            df = e['write_segments']
            # print(df)
            for line in df.values:
                time = (line[3] + line[2]) / 2
                idx = math.floor(time / step)
                # print(idx, time)
                write[i, idx] += 1
                write_bw[i, idx] += line[1]
    
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True, sharey='row')
    vmax = max(read.max(), write.max())
    vmax_bw = max(read_bw.max(), write_bw.max())
    extent = [0, int(duration.total_seconds()), 0, Y_size]
    output = (report.filename.split('/')[-1]).split('.')[-2]
    ax0.set_title('Reads')
    # ax0.set_xlabel('Time (s)')
    ax0.set_ylabel(sep)
    ax0.grid(True)
    im_read = ax0.imshow(read, cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax)
    fig.colorbar(im_read, ax=ax0)

    ax1.set_title('Writes')
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel(sep)
    ax1.grid(True)
    im_write = ax1.imshow(write, cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax)
    # fig.colorbar(im_write, ax=ax1)

    ax2.set_title('Reads Bandwidth')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(sep)
    ax2.grid(True)
    im_read_bw = ax2.imshow(read_bw, cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax_bw)
    fig.colorbar(im_read_bw, ax=ax2)

    ax3.set_title('Writes Bandwidth')
    ax3.set_xlabel('Time (s)')
    # ax3.set_ylabel(sep)
    ax3.grid(True)
    im_write_bw = ax3.imshow(write_bw, cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax_bw)
    # fig.colorbar(im_write_bw, ax=ax3)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle('I/O separated by {} as a function of time'.format(sep))
    plt.savefig('{}_dxt_posix_{}.png'.format(output, sep))
    # plt.show()
    # plt.close(fig)
    

def main ():
    argv = sys.argv
    if len(argv) < 2:
        print("Usage: python script.py <darshan log file>, see -h or --help for more information")
        exit(1)
    
    filename = argv[1]
    report = darshan.DarshanReport(filename, read_all=False)
    meta = report.metadata
    report.info()


    for i in range(2, len(argv)) :

        if argv[i] == '-h' or argv[i] == '--help' :
            print("Usage: python script.py <darshan log file> [options]")
            print("Options:")
            print("dxt_posix : Shows the I/O separated by file or rank, as a function of time")
            print("followed by : rank, file or all")
            exit(0)

        if argv[i] == 'dxt_posix' :
            if i + 1 >= len(argv) or not(argv[i+1] == 'File' or argv[i+1] == 'Rank' or argv[i+1] == 'all') :   
                dxt_posix_sorted_RW(report, 'Rank')
                dxt_posix_sorted_RW(report, 'File')
            else :
                dxt_posix_sorted_RW(report, argv[i+1])
                i += 1

main()