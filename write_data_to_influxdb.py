import pandas as pd
import datetime as dt

import influxdb_client
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

token = "4twszgEIHmgV0NfWQOkLHpIp-XfQoMXgiBELM5RlTWCdedWxTaX0ktqkfruSORou0s8vdYedkrNQqV2lSVqDVw=="
org = "isd"
url = "http://localhost:8086"
bucket="isd"

client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)


df = pd.read_parquet("data")
df["rtctime"] = pd.to_datetime(df["rtctime"], unit="ms")
colnames = df.columns

for i, row in df.iterrows():
    point = Point("sensor")
    for c in colnames[1:]:
        point = point.field(c, row[c])
    point.time(row["rtctime"], WritePrecision.MS)
    
    write_api.write(bucket=bucket, org=org, record=point)
    
    print(i)