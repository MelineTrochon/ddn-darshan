import pandas as pd

"""
This class shows in a tabular several information about the metadata operations that are not associated with I/O operations.
"""


class MetadataWithout_IO:
    def __init__(self, info):
        info.files[0].posix = pd.merge(
            info.files[0].posix["counters"],
            info.files[0].posix["fcounters"],
            on=["rank", "id"],
        )
        self.meta_posix = info.files[0].posix[
            (
                (info.files[0].posix["POSIX_READS"] == 0)
                & (info.files[0].posix["POSIX_WRITES"] == 0)
            )
        ]
        self.meta_posix = self.meta_posix.drop(
            self.meta_posix.columns[self.meta_posix.eq(0).all()], axis=1
        )

    def get_info(self, info):
        i = 0
        while i < len(self.meta_posix.columns):
            col = self.meta_posix.columns[i]
            if col != "id" and col != "rank" and col[0:8] != "POSIX_F_":
                temp_posix = self.meta_posix[self.meta_posix[col] != 0]
                temp_posix = temp_posix.drop(
                    temp_posix.columns[temp_posix.eq(0).all()], axis=1
                )
                self.meta_posix = self.meta_posix[self.meta_posix[col] == 0]
                self.meta_posix = self.meta_posix.drop(
                    self.meta_posix.columns[self.meta_posix.eq(0).all()], axis=1
                )
                summary = temp_posix.describe()
                sum_row = pd.DataFrame(
                    temp_posix.sum().values.reshape(1, -1),
                    columns=temp_posix.columns,
                    index=["sum"],
                )
                summary = pd.concat([summary, sum_row])
                summary.to_csv(info.output + "_meta_posix_" + col + ".csv")
            else:
                i += 1
