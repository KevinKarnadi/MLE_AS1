import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_gold_table(snapshot_date_str, 
                       silver_clickstream_directory, silver_attributes_directory, 
                       silver_financials_directory, silver_loan_daily_directory, 
                       gold_feature_store_directory, gold_label_store_directory, 
                       spark, dpd, mob):

    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Connect to silver tables
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df_clickstream = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df_clickstream.count())

    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df_attributes = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df_attributes.count())

    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df_financials = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df_financials.count())

    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df_labels = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df_labels.count())

    # Create gold label table
    # Get customer at mob
    df_labels = df_labels.filter(col("mob") == mob)
    # Get label
    df_labels = df_labels.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df_labels = df_labels.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))
    # Select columns to save
    df_labels = df_labels.select("loan_id", "Customer_ID", "label", "label_def", "loan_start_date", "snapshot_date")
    
    # Save gold label table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df_labels.write.mode("overwrite").parquet(filepath)
    # df_labels.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

    # Create gold feature table
    # Join the individual feature tables
    df_features = df_attributes.alias("attr") \
        .join(df_financials.alias("fin"),
            on=[col("attr.Customer_ID") == col("fin.Customer_ID"),
                col("attr.snapshot_date") == col("fin.snapshot_date")],
            how="left") \
        .join(df_clickstream.alias("click"),
            on=[col("attr.Customer_ID") == col("click.Customer_ID"),
                col("attr.snapshot_date") == col("click.snapshot_date")],
            how="left") \
        .select("attr.*",
                *[f"fin.{col}" for col in df_financials.columns if col not in ("Customer_ID", "snapshot_date")],
                *[f"click.{col}" for col in df_clickstream.columns if col not in ("Customer_ID", "snapshot_date")])
    
    # Save gold feature table - IRL connect to database to write
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df_features.write.mode("overwrite").parquet(filepath)
    # df_features.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)