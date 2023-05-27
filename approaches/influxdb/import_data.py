#import required packages
import pandas as pd
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import os

# influxdb config
token = "4twszgEIHmgV0NfWQOkLHpIp-XfQoMXgiBELM5RlTWCdedWxTaX0ktqkfruSORou0s8vdYedkrNQqV2lSVqDVw=="
org = "isd"
url = "http://localhost:8086"
bucket="isd"

# get client and write api
client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)


skip = 1 # use only every skip row
chunksize = 10000 # rows of dataframes to write at once
for fn in os.listdir("../data"):
    # filepath
    fp = os.path.join("../data", fn)
    
    # read data into pandas df       
    df = pd.read_parquet(fp)
    df = df.iloc[0:len(df):skip]
    df["rtctime"] = pd.to_datetime(df["rtctime"], unit="ms")
    df.set_index("rtctime", inplace=True)

    # write data in small chunks
    for i in range(0, len(df), chunksize):
        write_api.write(bucket, record=df.iloc[i:i+chunksize], data_frame_measurement_name=fn)
        print(f"\r{fn}: {i/len(df)*100: .2f}%", end="")
    print()
