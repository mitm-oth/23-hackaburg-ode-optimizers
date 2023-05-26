import pandas as pd


class Cutter():
    def __init__(self, data_path, gap_ms=10000, exclude_track_ms=2000):
        # * read parquet
        df = pd.read_parquet(data_path)
        df = df.sort_values("rtctime")
        assert (df.shape == (8159719, 10))

        # * find beginning of a record by finding gaps in rtctime
        deltas = df['rtctime'].diff()[1:]
        df["rtc_gap"] = deltas[deltas > gap_ms]

        # * add column is_track_beginning
        df["is_track_beginning"] = df['rtc_gap'].apply(
            lambda x: 1 if x > gap_ms else 0)
        df.at[0, "is_track_beginning"] = 1

        # * aggregate track_id based on rtc_gap
        df["track_id"] = df.is_track_beginning.cumsum(axis="index")

        # * dropping tracks that are smaller than 20s
        count_df = df.groupby(['track_id'])['track_id'].count()\
            .reset_index(name='count') \
            .sort_values(['count'], ascending=False)
        # short_track_ids = count_df[count_df["count"] <= exclude_track_ms].track_id.to_list()
        long_track_ids = count_df[count_df["count"]
                                  > exclude_track_ms].track_id.to_list()

        self.count_df = count_df
        self.df = df[df["track_id"].isin(long_track_ids)]

    def get_gen(self):
        current_id = 1
        while current_id < 200:
            next_df = self.df[self.df["track_id"] ==
                              current_id].sort_values("rtctime")
            current_id += 1
            if next_df.empty:
                continue
            yield next_df

    def get_biggest_track(self):
        track_id = self.count_df["track_id"].iloc[0]
        return self.df[self.df["track_id"] == track_id]


# for i in Cutter().get_gen():

# gen = Cutter().get_gen()
# df = next(gen)
