import darshan
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import array
import os


def merge_darshan_logs(path):
    """
    Merge multiple darshan logs from a given directory into a single DataFrame.

    Args:
        path (str): Directory path containing darshan log files.

    Returns:
        pd.DataFrame: Merged dxt_posix records DataFrame.
        int: Number of processes.
        pd.Timedelta: Duration of the job.
    """
    files = os.listdir(path)
    nprocs = len(files)
    dxt_posix = None
    start = None
    end = None

    for i, f in enumerate(files):
        if f.split('.')[-1] != 'darshan':
            print("The file {} is not a darshan log and will be ignored".format(f))
            continue
        report = darshan.DarshanReport(path + '/' + f, read_all=False)

        if 'DXT_POSIX' not in report.modules:
            print("The file {} does not contain dxt_posix records and will be ignored".format(f))
            continue

        report.read_all_dxt_records()
        report_dxt_posix = report.records['DXT_POSIX'].to_df()

        for line in report_dxt_posix:
            line['rank'] = i

        dxt_posix = dxt_posix.append(report_dxt_posix) if dxt_posix is not None else report_dxt_posix

        start = report.start_time if start is None or report.start_time < start else start
        end = report.end_time if end is None or report.end_time > end else end

    duration = end - start
    return dxt_posix, nprocs, duration


def create_sparse_matrices(sep, n, dxt_posix, step, Y_size):
    """
    Create sparse matrices for I/O operations and bandwidth based on the given separation.

    Args:
        sep (str): Separation criteria ('Rank' or 'File').
        n (int): Number of time intervals.
        dxt_posix (pd.DataFrame): Merged dxt_posix records DataFrame.
        step (float): Time interval step size.
        Y_size (int): Size of the Y-axis based on the separation criteria.

    Returns:
        sp.coo_matrix: Read operations matrix.
        sp.coo_matrix: Write operations matrix.
        sp.coo_matrix: Read bandwidth matrix.
        sp.coo_matrix: Write bandwidth matrix.
        sp.coo_matrix: Read bandwidth count matrix.
        sp.coo_matrix: Write bandwidth count matrix.
    """
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
        idx = int(line[3] / step)
        length = round((line[3] - line[2]) / 2) + 1
        loop = True

        if idx == idx_prec and i == i_prec + 1:
            v[i_prec] += 1
            v_bw[i_prec] += length
            loop = False

        if loop:
            idx_prec = idx
            i_prec = i
            x.append(int(line[0]))
            y.append(idx)
            v.append(1)
            v_bw.append(length)

        return i_prec, idx_prec

    for i, e in enumerate(dxt_posix):
        for line in e:
            if sep == 'Rank':
                x_read.append(int(line[0]))
                x_write.append(int(line[0]))
            elif sep == 'File':
                print(line)
                x_read.append(int(line[1]))
                x_write.append(int(line[1]))

        idx_prec_r, i_prec_r = incrementation(line, step, i, v_read, v_read_bw, x_read, y_read, i_prec_r, idx_prec_r)
        idx_prec_w, i_prec_w = incrementation(line, step, i, v_write, v_write_bw, x_write, y_write, i_prec_w, idx_prec_w)

    read_ops = sp.coo_matrix((v_read, (y_read, x_read)), shape=shape, dtype='i')
    write_ops = sp.coo_matrix((v_write, (y_write, x_write)), shape=shape, dtype='i')
    read_bw = sp.coo_matrix((v_read_bw, (y_read, x_read)), shape=shape, dtype='L')
    write_bw = sp.coo_matrix((v_write_bw, (y_write, x_write)), shape=shape, dtype='L')
    read_bw_count = sp.coo_matrix((v_read, (y_read, x_read)), shape=shape, dtype='i')
    write_bw_count = sp.coo_matrix((v_write, (y_write, x_write)), shape=shape, dtype='i')

    return read_ops, write_ops, read_bw, write_bw, read_bw_count, write_bw_count


def plot_heatmap(matrix, title, xlabel, ylabel, filename):
    """
    Plot a heatmap for the given matrix and save it to a file.

    Args:
        matrix (sp.coo_matrix): Sparse matrix to be visualized.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        filename (str): Output filename.
    """
    fig, ax = plt.subplots()
    heatmap = ax.imshow(matrix.toarray(), cmap='hot_r', interpolation='nearest', aspect='auto')
    fig.colorbar(heatmap)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.savefig(filename)
    plt.close()


def dxt_posix_sorted_IO(path, separation='Rank', intervals=10, output_dir='./output'):
    """
    Sort and plot I/O data based on the given separation criteria and time intervals.

    Args:
        path (str): Path to the darshan log files or directory.
        separation (str): Separation criteria ('Rank' or 'File'). Defaults to 'Rank'.
        intervals (int): Number of time intervals. Defaults to 10.
        output_dir (str): Directory to save the output plots. Defaults to './output'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.isdir(path):
        dxt_posix, nprocs, duration = merge_darshan_logs(path)
    elif os.path.isfile(path):
        report = darshan.DarshanReport(path, read_all=False)

        if 'DXT_POSIX' not in report.modules:
            print("The file {} does not contain dxt_posix records".format(path))
            return

        report.read_all_dxt_records()
        dxt_posix = report.records['DXT_POSIX'].to_df()
        nprocs = 1
        duration = report.end_time - report.start_time
    else:
        print("Invalid path: {}".format(path))
        return

    step = duration.total_seconds() / intervals
    # print(type(dxt_posix&))
    Y_size = nprocs if separation == 'Rank' else len(dxt_posix)

    read_ops, write_ops, read_bw, write_bw, read_bw_count, write_bw_count = create_sparse_matrices(separation, intervals, dxt_posix, step, Y_size)

    plot_heatmap(read_ops, 'Read Operations', 'Time Interval', separation, os.path.join(output_dir, 'read_ops.png'))
    plot_heatmap(write_ops, 'Write Operations', 'Time Interval', separation, os.path.join(output_dir, 'write_ops.png'))
    plot_heatmap(read_bw, 'Read Bandwidth', 'Time Interval', separation, os.path.join(output_dir, 'read_bw.png'))
    plot_heatmap(write_bw, 'Write Bandwidth', 'Time Interval', separation, os.path.join(output_dir, 'write_bw.png'))
    plot_heatmap(read_bw_count, 'Read Bandwidth Count', 'Time Interval', separation, os.path.join(output_dir, 'read_bw_count.png'))
    plot_heatmap(write_bw_count, 'Write Bandwidth Count', 'Time Interval', separation, os.path.join(output_dir, 'write_bw_count.png'))


# Example usage
dxt_posix_sorted_IO("gysela_biggest2.darshan", separation='File', intervals=20, output_dir='./output')
