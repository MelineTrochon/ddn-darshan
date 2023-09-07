# ddn-darshan

Tool to plot darshan logfile (with DXT enabled)

To run the script : 
$ python src/main.py <darshan log file> [feature] [its options] [second feature] [its options] ...
  
# Available features (by default, dxt_posix nb_rank_file and metadata are run):
    dxt_posix : Shows the I/O separated by file or rank, as a function of time

    nb_rank_file : Shows the number of rank per file at each time step
    
    metadata : Shows the metadata of the file that is not I/O related

    aggregate_info : Shows the aggregated I/O as a function of time (same as dxt_posix but it is plots instead of heatmaps)

# Available option for the features (if a feature is not available, it will skip the option and print a warning on stdout):

    change the output file : by default you have to create an output repository and the script will put all the output in it. 
        -output <output repository>

    the norm of the colorbar, by default it is linear, it uses matplotlib.pyplot normalisation option (example: linear, log, symlog, ...)
        -norm  <normalisation method>
    
    the number of bins (one time-block) for the heatmap, by default it is 50
        <number of bins>
    
    For dxt_posix, the sorting method (rank, file and hostname), by default it will do all of them
        <sorting method>
    
    For dxt_posix, the type of operation (read or write), by default it will do all of them
        <operation type>


# Example of a command line :
    python main.py <repository of darshan file> dxt_posix -output dxt_posix_output/ -norm log 100 rank file write aggregate_info 100 -output aggregate_info_output/ write

# Installation and utilisation command line :
    pip install miniconda
    pip create -n pydarshan
    pip activate pydarshan
    conda install python
    pip install darshan

    mkdir output
    python {path_to_the_script}/main.py <darshan log file> 