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
#print(df.head())
df["rtctime"] = pd.to_datetime(df["rtctime"], unit="ms")
df.set_index("rtctime", inplace=True)
#print(df.head())

write_api.write(bucket, record=df, data_frame_measurement_name="sensors")