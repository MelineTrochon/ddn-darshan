import darshan
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import time
import sys

def dxt_posix (report, sep='all') :
    if not 'DXT_POSIX' in report.records.keys() :
        print("The file does not contain dxt_posix records")
        exit(1)
    
    report.read_all_dxt_records()
    dxt_posix = report.records['DXT_POSIX'].to_df()
    duration = report.end_time - report.start_time
    n = 50
    step = int(duration.total_seconds()) / n
    X = np.linspace(0, int(duration.total_seconds()), n)
    read = np.zeros((len(dxt_posix), n))
    read_bw = np.zeros((len(dxt_posix), n))
    write = np.zeros((len(dxt_posix), n))
    write_bw = np.zeros((len(dxt_posix), n))
    extent = [0, int(duration.total_seconds()), 0, len(dxt_posix)]

    for i, e in enumerate(dxt_posix):
        if not e['read_segments'].empty:
            df = e['read_segments']
            for line in df.values:
                time = (line[3] + line[2]) / 2
                read[i, int(time / step)] += 1
                read_bw[i, int(time / step)] += line[1]
        if not e['write_segments'].empty:
            df = e['write_segments']
            for line in df.values:
                time = (line[3] + line[2]) / 2
                write[i, int(time / step)] += 1
                write_bw[i, int(time / step)] += line[1]
    
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    vmax = max(read.max(), write.max())
    vmax_bw = max(read_bw.max(), write_bw.max())
    if sep == 'file' or sep == 'all' :
        ax0.set_title('Reads')
        ax0.set_xlabel('Time (s)')
        ax0.set_ylabel('file')
        im_read = ax0.imshow(read, cmap='Reds', aspect='auto', extent=extent, origin='lower', vmax=vmax)
        fig.colorbar(im_read, ax=ax0)


def main ():
    argv = sys.argv
    if len(argv) < 2:
        print("Usage: python script.py <darshan log file>, see -h or --help for more information")
        exit(1)
    
    filename = argv[1]
    report = darshan.DarshanReport(filename)
    meta = report.metadata


    for i in range(2, len(argv)) :

        if argv[i] == '-h' or argv[i] == '--help' :
            print("Usage: python script.py <darshan log file> [options]")
            print("Options:")
            print("dxt_posix : Shows the I/O separated by file or rank, as a function of time")
            print("followed by : rank, file or all")
            exit(0)

        if argv[i] == 'dxt_posix' :
            if i + 1 >= len(argv) or not(argv[i+1] == 'file' or argv[i+1] == 'rank' or argv[i+1] == 'all') :   
                dxt_posix(report)
            else :
                dxt_posix(report, argv[i+1])
                i += 1

main()