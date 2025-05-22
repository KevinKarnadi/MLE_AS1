import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# Setup Date Config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# Generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

# Create bronze datalake
bronze_clickstream_directory = "datamart/bronze/feature_clickstream/"
clickstream_csv_file_path = "data/feature_clickstream.csv"
if not os.path.exists(bronze_clickstream_directory):
    os.makedirs(bronze_clickstream_directory)

bronze_attributes_directory = "datamart/bronze/features_attributes/"
attributes_csv_file_path = "data/features_attributes.csv"
if not os.path.exists(bronze_attributes_directory):
    os.makedirs(bronze_attributes_directory)

bronze_financials_directory = "datamart/bronze/features_financials/"
financials_csv_file_path = "data/features_financials.csv"
if not os.path.exists(bronze_financials_directory):
    os.makedirs(bronze_financials_directory)

bronze_lms_directory = "datamart/bronze/lms/"
lms_csv_file_path = "data/lms_loan_daily.csv"
if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

# Run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_clickstream_directory, clickstream_csv_file_path, "clickstream", spark)
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_attributes_directory, attributes_csv_file_path, "attributes", spark)
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_financials_directory, financials_csv_file_path, "financials", spark)
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_lms_directory, lms_csv_file_path, "loan_daily", spark)

# Create silver datalake
silver_clickstream_directory = "datamart/silver/feature_clickstream/"
if not os.path.exists(silver_clickstream_directory):
    os.makedirs(silver_clickstream_directory)

silver_attributes_directory = "datamart/silver/features_attributes/"
if not os.path.exists(silver_attributes_directory):
    os.makedirs(silver_attributes_directory)

silver_financials_directory = "datamart/silver/features_financials/"
if not os.path.exists(silver_financials_directory):
    os.makedirs(silver_financials_directory)

silver_loan_daily_directory = "datamart/silver/loan_daily/"
if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

# Run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table_clickstream(date_str, bronze_clickstream_directory, silver_clickstream_directory, "clickstream", spark)
    utils.data_processing_silver_table.process_silver_table_attributes(date_str, bronze_attributes_directory, silver_attributes_directory, "attributes", spark)
    utils.data_processing_silver_table.process_silver_table_financials(date_str, bronze_financials_directory, silver_financials_directory, "financials", spark)
    utils.data_processing_silver_table.process_silver_table_lms(date_str, bronze_lms_directory, silver_loan_daily_directory, "loan_daily", spark)

# Create gold datalake
gold_feature_store_directory = "datamart/gold/feature_store/"
if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

gold_label_store_directory = "datamart/gold/label_store/"
if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# Run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_gold_table(date_str, 
                                                        silver_clickstream_directory, silver_attributes_directory, 
                                                        silver_financials_directory, silver_loan_daily_directory, 
                                                        gold_feature_store_directory, gold_label_store_directory, 
                                                        spark, dpd=30, mob=6)