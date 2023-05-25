#import required packages
import pandas as pd
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

# influxdb config
token = "4twszgEIHmgV0NfWQOkLHpIp-XfQoMXgiBELM5RlTWCdedWxTaX0ktqkfruSORou0s8vdYedkrNQqV2lSVqDVw=="
org = "isd"
url = "http://localhost:8086"
bucket="isd"

# get client and write api
client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

# read data into pandas df       
df = pd.read_parquet("../data")
df["rtctime"] = pd.to_datetime(df["rtctime"], unit="ms")
df.set_index("rtctime", inplace=True)

# write data in small chunks
step = 10000
for i in range(0, len(df), step):
    print(f"\rprogress: {i/len(df): .2f}", end="")
    write_api.write(bucket, record=df.iloc[i:i+step], data_frame_measurement_name="sensors")