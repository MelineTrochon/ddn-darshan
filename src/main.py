import sys
import time

from DarshanInfo import DarshanInfo


"""
This is the main file that is used to run the different functions of DarshanInfo.
"""

def usage():
    print("Usage: python darshan-info.py <Darshan file or repository> [Options]")
    print("If no options, then all the options will be done")
    print("Options :")
    print("dxt_posix : Shows the I/O grouped by file or rank, as a function of time")
    print("nb_rank_file : Shows the number of rank that access the same file each time step")
    print("metadata : Shows the metadata of the application that are not I/O related")
    print("To see the options, check the comment of each python file.")

    return


def main():

    argv = sys.argv
    if len(argv) < 2 or argv[1] == '-h' or argv[1] == '--help':
        usage()
        return

    info = DarshanInfo(argv)

    info.i = 2
    if len(argv) == 2 :
        start_t = time.time()
        info.dxt_posix_heatmap()
        print("dxt_posix_heatmap took %.3f seconds"%(time.time() - start_t))

        start_t = time.time()
        info.nb_rank_file()
        print("nb_rank_file took %.3f seconds"%(time.time() - start_t))

        start_t = time.time()
        info.metadata_without_IO()
        print("metadata took %.3f seconds"%(time.time() - start_t))
        return

    while info.i < len(argv):

        if argv[info.i] == 'dxt_posix':
            start_t = time.time()
            info.dxt_posix_heatmap()
            print("dx_posix_heatmap took %.3f seconds"%(time.time() - start_t))
            continue  
        
        if argv[info.i] == 'nb_rank_file':
            start_t = time.time()
            info.nb_rank_file()
            print("nb_rank_file took %.3f seconds"%(time.time() - start_t))
            continue

        if argv[info.i] == 'metadata':
            start_t = time.time()
            info.metadata_without_IO()
            print("metadata took %.3f seconds"%(time.time() - start_t))
            continue

        else:
            print("Option {} is not recognized".format(argv[info.i]))
            usage()
            return


if __name__ == "__main__":
    main()