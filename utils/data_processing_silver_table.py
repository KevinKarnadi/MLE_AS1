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

from pyspark.sql.functions import col, when, regexp_replace, regexp_extract, split, udf, ceil, datediff, add_months, lit, explode, split, trim
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import re

def process_silver_table_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, table_name, spark):
    
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Connect to bronze table
    partition_name = "bronze" + "_"+table_name+"_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "fe_1": IntegerType(),
        "fe_2": IntegerType(),
        "fe_3": IntegerType(),
        "fe_4": IntegerType(),
        "fe_5": IntegerType(),
        "fe_6": IntegerType(),
        "fe_7": IntegerType(),
        "fe_8": IntegerType(),
        "fe_9": IntegerType(),
        "fe_10": IntegerType(),
        "fe_11": IntegerType(),
        "fe_12": IntegerType(),
        "fe_13": IntegerType(),
        "fe_14": IntegerType(),
        "fe_15": IntegerType(),
        "fe_16": IntegerType(),
        "fe_17": IntegerType(),
        "fe_18": IntegerType(),
        "fe_19": IntegerType(),
        "fe_20": IntegerType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Save silver table - IRL connect to database to write
    partition_name = "silver" + "_"+table_name+"_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

# Spark UDF for string pattern matching
def ssn_check(ssn):
    return ssn if re.fullmatch(r"\d{3}-\d{2}-\d{4}", str(ssn)) else "None"
ssn_check_udf = udf(ssn_check, StringType())

def process_silver_table_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, table_name, spark):
    
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Connect to bronze table
    partition_name = "bronze" + "_"+table_name+"_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Clean data: handle bad formatting
    df = df.withColumn("Age", regexp_replace(col("Age").cast(StringType()), "_", ""))
    df = df.withColumn("SSN", ssn_check_udf(col("SSN")))
    df = df.withColumn("Occupation", when(col("Occupation") == "_______", "None").otherwise(col("Occupation")))

    # Clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
    
    # Augment data: Add "None" indicator columns
    df = df.withColumn("No_SSN", when(col("SSN") == "None", 1).otherwise(0))
    df = df.withColumn("No_Occupation", when(col("Occupation") == "None", 1).otherwise(0))

    # Save silver table - IRL connect to database to write
    partition_name = "silver" + "_"+table_name+"_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_table_financials(snapshot_date_str, bronze_financials_directory, silver_financials_directory, table_name, spark):
    
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Connect to bronze table
    partition_name = "bronze" + "_"+table_name+"_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Clean data: handle bad formatting
    df = df.withColumn("Annual_Income", regexp_replace(col("Annual_Income").cast(StringType()), "_", ""))
    df = df.withColumn("Num_of_Loan", regexp_replace(col("Num_of_Loan").cast(StringType()), "_", ""))
    df = df.withColumn("Changed_Credit_Limit", when(col("Changed_Credit_Limit") == "_", 0).otherwise(col("Changed_Credit_Limit")))
    df = df.withColumn("Credit_Mix", when(col("Credit_Mix") == "_", "Unknown").otherwise(col("Credit_Mix")))
    df = df.withColumn("Outstanding_Debt", regexp_replace(col("Outstanding_Debt").cast(StringType()), "_", ""))
    df = df.withColumn("Amount_invested_monthly", regexp_replace(col("Amount_invested_monthly").cast(StringType()), "_", ""))
    df = df.withColumn("Monthly_Balance", col("Monthly_Balance").cast("double"))
    df = df.fillna({"Monthly_Balance": 0.0})

    # Clean data: handle missing values
    df = df.fillna({"Type_of_Loan": "Not Specified"})

    # Clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),  # extract
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": FloatType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),  # extract
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),  # extract
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Feature extraction from string columns

    # Extract loan types from "Type_of_Loan" column
    loan_types = ['Credit-Builder Loan', 'Home Equity Loan', 'Mortgage Loan', 'Auto Loan', 
                  'Personal Loan', 'Student Loan', 'Payday Loan', 'Debt Consolidation Loan']
    for loan in loan_types:
        df = df.withColumn(loan, col("Type_of_Loan").contains(loan))
    
    # Convert "Credit_History_Age" column to number of months
    def credit_age_to_months(s):
        if s and 'and' in s:
            y, m = re.findall(r'(\d+)', s)
            return int(y) * 12 + int(m)
        return 0
    credit_udf = udf(credit_age_to_months, IntegerType())
    df = df.withColumn("Credit_History_Age_Months", credit_udf(col("Credit_History_Age")))

    # Extract spent & value from Payment_Behaviour
    df = df.withColumn("Payment_Behaviour_Spent", regexp_extract(col("Payment_Behaviour"), r"^(.*?)_spent_", 1))
    df = df.withColumn("Payment_Behaviour_Value", regexp_extract(col("Payment_Behaviour"), r"_spent_(.*?)_value_payments$", 1))

    # Save silver table - IRL connect to database to write
    partition_name = "silver" + "_"+table_name+"_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_table_lms(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, table_name, spark):
    
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Connect to bronze table
    partition_name = "bronze" + "_"+table_name+"_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Augment data: Add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # Augment data: Add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # Save silver table - IRL connect to database to write
    partition_name = "silver" + "_"+table_name+"_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df