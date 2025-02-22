# Databricks notebook source
from pyspark.sql import * 
from pyspark.sql.functions import * 
from pyspark.sql.types import *

# COMMAND ----------

display(dbutils.fs.ls('/FileStore/'))

# COMMAND ----------

df=spark.read.format('csv').option('header',True).option('inferSchema',True).load('dbfs:/FileStore/Raw_data.csv')

# COMMAND ----------

#Check columns & data type of file 
df.printSchema()

# COMMAND ----------

# Check MB Data in brand column
df.filter(df.Brand.isin(['MB','Mercedes','Mercedes Benz'])).count()

# COMMAND ----------

#Check PC in Business unit column

df.filter(df['Business Unit'].isin(['PC','Cars','Passenger','Passenger Cars'])).count()

# COMMAND ----------

# Check month data in month Number Column 
print(df.filter(df['Month Number'].isin([1])).count())
print(df.filter(df['Month Number'].isin([2])).count())
print(df.filter(df['Month Number'].isin([3])).count())

# COMMAND ----------

#Convert data  to lowercase in market column

df=df.withColumn('Market',initcap(df['Market']))

# COMMAND ----------

df.display()

# COMMAND ----------

# Modify data in brand column
df=df.withColumn('Brand',when(df['Brand']=='MB',"Mercedes Benze").when(df['Brand']=='Mercedes','Mercedes Benze').when(df['Brand']=='MBK','Mercedes Benze').otherwise(df['Brand']))
df.display()

# COMMAND ----------

# Modify data in BU column
df=df.withColumn('Business Unit',when(df['Business Unit']=='PC',"Passenger Cars").when(df['Business Unit']=='Cars','Passenger Cars').when(df['Business Unit']=='Minivan','Vans').otherwise(df['Business Unit']))
df.display()

# COMMAND ----------

from pyspark.sql.types import IntegerType

# COMMAND ----------

# Change data type of Gross & Net spend column
df=df.withColumn('Local Cost Net Spend',df['Local Cost Net Spend'].cast(IntegerType())).withColumn('Local Cost Gross Spend',df['Local Cost Gross Spend'].cast(IntegerType()))

# COMMAND ----------

df.write.format('parquet').save('dbfs:/FileStore/output/Processed data.parquet')

# COMMAND ----------

df1=spark.read.format('parquet').load('dbfs:/FileStore/output/Processed data.parquet')

# COMMAND ----------

df1.display()
