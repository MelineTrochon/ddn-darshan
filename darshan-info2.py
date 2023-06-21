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
    print("\tit can be followed by : Rank, File or All, default is all")
    return


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
        dxt_posix = dxt_posix.append(report_dxt_posix) if dxt_posix is not None else report_dxt_posix

        start = report_start if start is None or report_start < start else start
        end = report_end if end is None or report_end > end else end

    return dxt_posix, nprocs, start, end


def dxt_posix_heatmap(i, argv):
    groups = {'rank', 'file'}
    n=50
    path = argv[1]
    t = len(argv)
    output = path.split('/')[-1]
    out = False

    for t in range(i+1, len(argv)):
        
        if out:
            output = argv[t]
            continue

        if argv[t] in groups:
            groups = {argv[t]}
            continue
        
        if argv[t].isdigit() :
            n = argv[t]
            continue
        
        if argv[t] == '-o':
            out = True
            continue
        
        break
        
    print("="*100 + "\nStart reading dxt records sorted by {}\n".format(groups))

    start_t = time.time()
    if os.path.isdir(path):
        dxt_posix, nprocs, start, end = merge_darshan_logs(path)
    
    elif path.split('.')[-1] == 'darshan' :
        dxt_posix, nprocs, start, end = darshan_file(path)
    
    else :
        print("The given path could not be read")
        exit(0)
    print("Read and convert all the data in %.3f sec"%(time.time() - start_t))

    step = (start - end).total_seconds() / n
    for group in groups :
        Y_size = nprocs if group ==  'Rank' else len(dxt_posix)
        plot_dxt_posix_heatmap(dxt_posix, group, n, step, Y_size, output)
    return t


def plot_dxt_posix_heatmap(dxt_posix, group, n, step, Y_size, output):

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

    def incrementation(line, step, i, v, v_bw, x, y, i_prec, idx_prec):
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
        if group == 'Rank' :
            i = e['rank']
        if group == 'File' :
            i = it
        if not e['read_segments'].empty:
            df = e['read_segments']
            for line in df.values:
                incrementation(line, step, i, v_read, v_read_bw, x_read, y_read, i_prec_r, idx_prec_r)

        if not e['write_segments'].empty:
            df = e['write_segments']
            for line in df.values:
                incrementation(line, step, i, v_write, v_write_bw, x_write, y_write, i_prec_w, idx_prec_w)

    def sparse_matrix(x, y, v, v_bw, shape=shape):
        mat = sp.coo_matrix((v, (x, y)), shape=shape)
        mat_bw = sp.coo_matrix((v_bw, (x, y)), shape=shape)
        mat.sum_duplicates()
        mat_bw.sum_duplicates()

        v_bw_count = mat_bw.data / mat.data
        mat_bw_count = sp.coo_matrix((v_bw_count, (x, y)), shape=shape)
        return mat, mat_bw, mat_bw_count

    read, read_bw, read_bw_count = sparse_matrix(x_read, y_read, v_read, v_read_bw)
    write, write_bw, write_bw_count = sparse_matrix(x_write, y_write, v_write, v_write_bw)

    plot_heatmap(read)
    plot_heatmap(write)
    plot_heatmap(read_bw)
    plot_heatmap(write_bw)
    plot_heatmap(read_bw_count)
    plot_heatmap(v_write_bw_count)

def plot_heatmap():
    return


def main():
    argv = sys.argv
    if len(argv) < 2 or argv[1] == '-h' or argv[1] == '--help':
        usage()
        return 0
    
    if len(argv) == 2:
        dxt_posix_heatmap(2, argv)

    i = 2
    while i < len(argv):

        if argv[i] == 'dxt_posix' :
            start_t = time.time()
            i = dxt_posix_heatmap(i, argv)
            print("dxt_posix in %.3f sec"%(time.time() - start_t))
        
        else : 
            print("The option " + str(argv[i]) + " is not recognized")

        i += 1

main()