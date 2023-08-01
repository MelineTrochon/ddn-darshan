import time
import os
from DarshanFile import DarshanFile
from RW_SparseMatrix import RW_SparseMatrix
from FileNbRankPerSec import FileNbRankPerSec
from MetadataWithout_IO import MetadataWithout_IO
from AggregateInfo import AggregateInfo

"""
This class is used as a container for all the darshan files in a directory (through the class DarshanFile)
It, then, redirects to the class that was called by the user (e.g. RW_SparseMatrix, FileNbRankPerSec, MetadataWithout_IO)
"""


class DarshanInfo:
    def __init__(self, argv):
        self.path = argv[1]
        self.argv = argv
        self.load_darshan_files()

    def load_darshan_files(self):
        print("=" * 100 + "\nStart reading darshan files\n")
        start_t = time.time()

        if os.path.isdir(self.path):
            self.nprocs = 0
            self.start = None
            self.end = None

            filenames = os.listdir(self.path)
            self.files = list()
            for it, filename in enumerate(filenames):
                if filename.split(".")[-1] != "darshan":
                    print("The file {} is not a darshan file".format(filename))
                    continue

                f = DarshanFile(self.path + "/" + filename)
                self.files.append(f)
                self.nprocs += f.nprocs
                self.start = (
                    f.report.start_time
                    if not self.start or self.start > f.report.start_time
                    else self.start
                )
                self.end = (
                    f.report.end_time
                    if not self.end or self.end < f.report.end_time
                    else self.end
                )

        elif self.path.split(".")[-1] == "darshan":
            self.files = [DarshanFile(self.path)]
            self.nprocs = self.files[0].nprocs
            self.start = self.files[0].report.start_time
            self.end = self.files[0].report.end_time

        else:
            print("Couldn't read the file {}".format(self.path))
            sys.exit(1)

        self.duration = (self.end - self.start).total_seconds() + 1
        print("Read the Darshan files in %.3f sec\n" % (time.time() - start_t))
        return

    def dxt_posix_heatmap(self):
        self.i += 1
        self.nbins = 50
        self.output = "output/" + self.path.split("/")[-1].split(".")[0]
        self.norm = "linear"
        self.groups = list()
        self.all_groups = {"rank", "file", "hostname"}
        self.ops = list()
        self.all_ops = {"read", "write"}

        while self.i < len(self.argv):
            arg = self.argv[self.i]
            if arg == "-output":
                self.output = self.argv[self.i + 1]
                self.i += 2
                continue

            if arg == "-norm":
                self.norm = self.argv[self.i + 1]
                self.i += 2
                continue

            if arg.isdigit():
                self.nbins = int(arg)
                self.i += 1
                continue

            if arg in self.all_groups:
                self.groups += {arg}
                self.i += 1
                continue

            if arg in self.all_ops:
                self.ops += {arg}
                self.i += 1
                continue

            break

        if len(self.groups) == 0:
            self.groups = self.all_groups

        if len(self.ops) == 0:
            self.ops = self.all_ops

        print(
            "=" * 100
            + "\nStart {} dxt_posix_heatmap sorted by {}\n".format(
                self.ops, self.groups
            )
        )

        start_t = time.time()
        self.len_dxt_posix = 0
        self.len_posix = 0
        self.hostnames = set()
        self.file_ids = set()
        for f in self.files:
            # print(f.p)
            self.len_dxt_posix += f.get_dxt_posix()
            self.len_posix += f.get_posix()
            for e in f.dxt_posix:
                self.hostnames.add(e["hostname"])
                self.file_ids.add(e["id"])
        self.hostnames = list(self.hostnames)
        self.file_ids = list(self.file_ids)

        print("Read and convert all data in %.3f sec\n" % (time.time() - start_t))

        self.step = self.duration / self.nbins
        self.matrix = list()
        for group in self.groups:
            for op in self.ops:
                self.group = group
                self.op = op
                print(
                    "=" * 50
                    + "\nStart generating {} heatmap sorted by {}".format(op, group)
                )

                start_t = time.time()
                self.matrix.append(RW_SparseMatrix(self))
                print("Generate sparse matrix in %.3f sec" % (time.time() - start_t))

                start_t = time.time()
                self.matrix[-1].to_heatmap(self)
                print("Generate heatmap in %.3f sec" % (time.time() - start_t))
        return

    def nb_rank_file(self):
        self.i += 1
        self.nbins = 50
        self.output = "output/" + self.path.split("/")[-1].split(".")[0]
        self.norm = "linear"
        self.ops = list()
        self.all_ops = {"read", "write"}

        while self.i < len(self.argv):
            arg = self.argv[self.i]
            if arg == "-output":
                self.output = self.argv[self.i + 1]
                self.i += 2
                continue

            if arg == "-norm":
                self.norm = self.argv[self.i + 1]
                self.i += 2
                continue

            if arg.isdigit():
                self.nbins = int(arg)
                self.i += 1
                continue

            if arg in self.all_ops:
                self.ops += {arg}
                self.i += 1
                continue

            break

        if len(self.ops) == 0:
            self.ops = self.all_ops

        print("=" * 100 + "\nStart {} number of rank per file\n".format(self.ops))

        start_t = time.time()
        self.len_dxt_posix = 0
        self.len_posix = 0
        self.hostnames = set()
        self.file_ids = set()
        for f in self.files:
            self.len_dxt_posix += f.get_dxt_posix()
            # self.len_posix += f.get_posix()
            for e in f.dxt_posix:
                self.hostnames.add(e["hostname"])
                self.file_ids.add(e["id"])
        self.hostnames = list(self.hostnames)
        self.file_ids = list(self.file_ids)

        print("Read and convert all data in %.3f sec\n" % (time.time() - start_t))

        self.step = self.duration / self.nbins
        self.matrix = list()
        for op in self.ops:
            self.op = op
            print("=" * 50 + "\nStart generating {} heatmap".format(op))

            start_t = time.time()
            self.matrix.append(FileNbRankPerSec(self))
            print("Generate sparse matrix in %.3f sec" % (time.time() - start_t))

            start_t = time.time()
            self.matrix[-1].to_heatmap(self)
            print("Generate heatmap in %.3f sec" % (time.time() - start_t))
        return

    def metadata_without_IO(self):
        self.i += 1
        self.output = "output/" + self.path.split("/")[-1].split(".")[0]

        print()
        print("=" * 100 + "\nStart generating metadata without IO\n")

        start_t = time.time()
        self.len_posix = 0
        for f in self.files:
            self.len_posix += f.get_posix()
        print("Read and convert all data in %.3f sec\n" % (time.time() - start_t))

        start_t = time.time()
        metadata = MetadataWithout_IO(self)
        metadata.get_info(self)
        print("Generate metadata in %.3f sec" % (time.time() - start_t))

    def aggregate_info(self):
        self.i += 1
        self.output = "output/" + self.path.split("/")[-1].split(".")[0]
        self.nbins = 50
        self.norm = "linear"
        self.ops = list()
        self.all_ops = {"read", "write"}

        while self.i < len(self.argv):
            arg = self.argv[self.i]
            if arg == "-output":
                self.output = self.argv[self.i + 1]
                self.i += 2
                continue

            if arg == "-norm":
                self.norm = self.argv[self.i + 1]
                self.i += 2
                continue

            if arg.isdigit():
                self.nbins = int(arg)
                self.i += 1
                continue

            if arg in self.all_ops:
                self.ops += {arg}
                self.i += 1
                continue

            break

        if len(self.ops) == 0:
            self.ops = self.all_ops

        print("=" * 100 + "\nStart {} aggregate info\n".format(self.ops))

        start_t = time.time()
        self.len_dxt_posix = 0
        self.len_posix = 0
        self.hostnames = set()
        self.file_ids = set()
        for f in self.files:
            self.len_dxt_posix += f.get_dxt_posix()
            self.len_posix += f.get_posix()
            for e in f.dxt_posix:
                self.hostnames.add(e["hostname"])
                self.file_ids.add(e["id"])
        self.hostnames = list(self.hostnames)
        self.file_ids = list(self.file_ids)

        print("Read and convert all data in %.3f sec\n" % (time.time() - start_t))

        self.step = self.duration / self.nbins
        self.matrix = list()
        for op in self.ops:
            self.op = op
            print("=" * 50 + "\nStart generating {} heatmap".format(op))

            start_t = time.time()
            self.matrix.append(AggregateInfo(self))
            print("Generate sparse matrix in %.3f sec" % (time.time() - start_t))

            start_t = time.time()
            self.matrix[-1].to_plot(self)
            print("Generate heatmap in %.3f sec" % (time.time() - start_t))

        return
