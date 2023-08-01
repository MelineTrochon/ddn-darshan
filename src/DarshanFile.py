import darshan
import time

"""
This class is used to read a Darshan file and extract the information, it is used by DarshanInfo.
"""


class DarshanFile:
    def __init__(self, f):
        self.file = f.split("/")[-1]
        self.report = darshan.DarshanReport(f, read_all=False)
        self.nprocs = self.report.metadata["job"]["nprocs"]
        self.dxt_posix = None
        self.posix = None

    def get_dxt_posix(self):
        if "DXT_POSIX" not in self.report.modules:
            print("The file {} does not have DXT_POSIX records".format(self.file))
            return 0

        if "DXT_POSIX" in self.report.records:
            print("The file {} has already loaded DXT_POSIX records".format(self.file))

        else:
            print("Reading DXT_POSIX in the file {}".format(self.file), end="\t")
            start_t = time.time()
            self.report.read_all_dxt_records()
            print("in %.3f sec" % (time.time() - start_t), end="\t")

        if self.dxt_posix:
            print(
                "The file {} has already converted DXT_POSIX records".format(self.file)
            )
            return len(self.dxt_posix)

        start_t = time.time()
        self.dxt_posix = self.report.records["DXT_POSIX"].to_df()
        print("Convert DXT_POSIX to dataframe in %.3f sec" % (time.time() - start_t))
        return len(self.dxt_posix)

    def get_posix(self):
        if "POSIX" not in self.report.modules:
            print("The file {} does not have POSIX records".format(self.file))
            return

        if "POSIX" in self.report.records:
            print("The file {} has already loaded POSIX records".format(self.file))

        else:
            print("Reading POSIX in the file {}".format(self.file), end="\t")
            start_t = time.time()
            self.report.read_all_generic_records()
            print("in %.3f sec" % (time.time() - start_t), end="\t")

        if self.posix:
            print("The file {} has already converted POSIX records".format(self.file))
            return len(self.posix)

        start_t = time.time()
        self.posix = self.report.records["POSIX"].to_df()
        print("Convert POSIX to dataframe in %.3f sec" % (time.time() - start_t))
        return len(self.posix)
