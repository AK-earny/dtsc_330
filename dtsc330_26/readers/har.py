"""
Read in Human Activity Recognition (HAR) data for classification of sleep
"""

import os

import pandas as pd


class HAR:
    def __init__(self, path: str, n_people: int):
        """Take in a path, read in the data files

        Args:
            path (str): the BASE path of the directory
            n_people (int): the number of people's data to read
        """
        hrs = self._read_hr(path, n_people)
        mots = self._read_motion(path, n_people)
        lbls = self._read_labels(path, n_people)
        self.df = self._combine(hrs, mots, lbls)

    def _read_hr(self, path: str, n_people: int = 1):
        """Read the heartrate from the base path"""
        path = os.path.join(path, "heart_rate")
        people = os.listdir(path)

        dfs = []
        count = 0
        for person in people:
            hrdf = pd.read_csv(
                os.path.join(path, person), names=["timestamp", "hr"]
            ).drop_duplicates("timestamp", keep="first")
            hrdf["person"] = count
            dfs.append(hrdf)
            count = count + 1  # count += 1
            if count >= n_people:
                break

        return pd.concat(dfs)

    def _read_motion(self, path: str, n_people: int = 1):
        """Read the motion from the base path"""
        path = os.path.join(path, "motion")
        people = os.listdir(path)

        dfs = []
        count = 0
        for person in people:
            motdf = pd.read_csv(
                os.path.join(path, person),
                delimiter=" ",
                names=["timestamp", "acc_x", "acc_y", "acc_z"],
            ).drop_duplicates("timestamp", keep="first")
            motdf["person"] = count
            dfs.append(motdf)
            count = count + 1  # count += 1
            if count >= n_people:
                break

        return pd.concat(dfs)

    def _read_labels(self, path: str, n_people: int = 1):
        """Read the heartrate from the base path"""
        path = os.path.join(path, "labels")
        people = os.listdir(path)

        dfs = []
        count = 0
        for person in people:
            print(person)
            labdf = pd.read_csv(
                os.path.join(path, person), names=["timestamp", "label"], delimiter=" "
            ).drop_duplicates("timestamp", keep="first")
            labdf["person"] = count
            dfs.append(labdf)
            count = count + 1  # count += 1
            if count >= n_people:
                break

        comb = pd.concat(dfs)
        comb["is_sleep"] = comb["label"] == 0
        return comb.loc[comb["label"] > -1].copy()  # This excludes missing data

    def _combine(self, hrs: pd.DataFrame, mots: pd.DataFrame, lbls: pd.DataFrame):
        """Combine three dataframes by interpolating to the highest sampling rate.
        The problem that we were having in class is that the timestamps were
        collected separately. There are multiple ways to combine them, but this
        leaves the data closest to its raw state.
        """
        # In the case of multiple people, they may have been recorded
        # with some overlap. We will fix that by using a recursive
        # function-- a function that calls itself.
        if len(pd.unique(hrs["person"])) > 1:
            people = pd.unique(hrs["person"])
            out = []
            for person in people:
                out.append(
                    self._combine(
                        hrs.loc[hrs["person"] == person],
                        mots.loc[mots["person"] == person],
                        lbls.loc[lbls["person"] == person],
                    )
                )
            return pd.concat(out)

        # Calculate median time interval between consecutive points for
        # each dataframe
        # We will also exclude time points that aren't in all three dataframes
        min_interval = None
        min_interval_df = None
        last_start = None
        first_end = None
        dfs = {"hrs": hrs, "mots": mots, "lbls": lbls}
        sampling_intervals = {}
        for name, df in dfs.items():
            df_sorted = df.sort_values("timestamp")
            time_diffs = df_sorted["timestamp"].diff().dropna()
            sampling_interval = time_diffs.median()

            if min_interval is None or sampling_interval < min_interval:
                min_interval = sampling_interval
                min_interval_df = name

            # Get the bounds of the recording time
            start_time = df_sorted["timestamp"].min()
            end_time = df_sorted["timestamp"].max()
            if last_start is None or start_time > last_start:
                last_start = start_time
            if first_end is None or end_time < first_end:
                first_end = end_time

        # Prepare each dataframe with timestamp as index
        for name in dfs:
            dfs[name] = dfs[name].copy()  # Copying prevents downstream errors
            dfs[name] = dfs[name].set_index("timestamp").sort_index()

        # Use the timestamps from the highest-frequency dataframe,
        # filtered to the time range where all three dataframes have data
        shared_index = dfs[min_interval_df].index
        shared_index = shared_index[
            (shared_index >= last_start) & (shared_index <= first_end)
        ]

        # Interpolate each dataframe to the shared index
        # This is going to look VERY confusing. Let's walk through it.
        # 1. dfs['hrs'].reindex(
        # We're going to change the index of the dataframe.
        # 2. dfs['hrs'].index.union(shared_index))
        # This might look unnecessary, but we're making an index that
        # includes BOTH the shared_index as well as the one from the
        # original dataframe. If we didn't and the timestamps didn't
        # perfectly match, we would end up with no data after using the
        # shared_index (because the timestamps specific to hrs wouldn't
        # exist within the timestamps for mots, which is the highest
        # frequency).
        # 3. .interpolate(method='index')
        # This computes the values in between by smoothly estimating them
        # .ffill fills them in forwards, which is appropriate for
        # categorical data
        # 4. .loc[shared_index]
        # Only keep the indices from mots.

        hrs_interp = (
            dfs["hrs"]
            .reindex(dfs["hrs"].index.union(shared_index))
            .interpolate(method="index")
            .loc[shared_index]
        )
        mots_interp = (
            dfs["mots"]
            .reindex(dfs["mots"].index.union(shared_index))
            .interpolate(method="index")
            .loc[shared_index]
        )
        lbls_interp = (
            dfs["lbls"]
            .reindex(dfs["lbls"].index.union(shared_index))
            .ffill()
            .loc[shared_index]
        )

        # Combine the dataframes
        combined = (
            pd.concat(
                [
                    hrs_interp[["hr", "person"]],
                    mots_interp[["acc_x", "acc_y", "acc_z"]],
                    lbls_interp[["is_sleep"]],
                ],
                axis=1,
            )
            .dropna()
            .reset_index()
        )  # Remove NaNs and fix the index

        return combined

