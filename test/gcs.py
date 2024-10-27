import streamlit as st
import pandas as pd
import gcsfs
from google.cloud import storage
from st_files_connection import FilesConnection

st.markdown("### This is a test of loading data from GCS")

# Initialize a GCS client
client = storage.Client()

# Create connection object and retrieve file contents.
# Specify input format is a csv and to cache the result for 600 seconds.
conn = st.connection('gcs', type=FilesConnection)
df = conn.read("sami-streamlit-bucket/yfdownload.csv", input_format="csv", ttl=600)

# Print results.
st.dataframe(data=df)



