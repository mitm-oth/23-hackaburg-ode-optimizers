import pandas as pd

ms_gap = 10000


class Cutter():
    def __init__(self, data_folder):
        # * read parquet

        df = pd.read_parquet(data_folder)
        df = df.sort_values("rtctime")
        assert (df.shape == (8159719, 10))

        # * find beginning of a record by finding gaps in rtctime

        deltas = df['rtctime'].diff()[1:]

        df["rtc_gap"] = deltas[deltas > ms_gap]

        # * add column is_track_beginning

        df["is_track_beginning"] = df['rtc_gap'].apply(
            lambda x: 1 if x > ms_gap else 0)

        df.at[0, "is_track_beginning"] = 1

        # * aggregate track_id based on rtc_gap

        df["track_id"] = df.is_track_beginning.cumsum(axis="index")

        # * dropping tracks that are smaller than 20s
        count_df = df.groupby(['track_id'])['track_id'].count()\
            .reset_index(name='count') \
            .sort_values(['count'], ascending=False)
        # short_track_ids = count_df[count_df["count"] <= 2000].track_id.to_list()
        long_track_ids = count_df[count_df["count"] > 2000].track_id.to_list()

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


# for i in Cutter().get_gen():

# gen = Cutter().get_gen()
# df = next(gen)
