from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *

############################################################################ DBFS##################################################################################

dbutils.help()
dbutils.widget.help()
dbutils.fs.help()                                             : file system
%fs ls /filestore/                                            : Enlist of file
display(dbutils.fs.ls('/filesore/))                           : Enlist 
dbutils.fs.mkdirs('filestore/table1')                         : create directory 
dbutils.fs.put('/filesore/file.txt,'hello there ')            : add file  with text 
dbutils.fs.head('/filesore/file.txt)                          :  read first rows
dbutils.fs.rm('/filesore/' ,true )                            : remove folder
dbutils.notebook.run('/filesore/' ,60 )                       :(url,time): run notebook
dbutils.notebook,exit("Some parameter values")                :exit  notebook
%run ./file path                                              : run notebook
dbutils.fs.cp('Source url','target url',true)                 :copy file , folder                                            
dbutils.fs.mv('/filstore/','filestore/prashant/',true).       :Move file folder
%run "./Includes/Classroom-Setup".                            : Class room setup
%run "./Includes/Classroom-Cleanup".                          : Classroom cleanu

################################################## WIDGETS ###############
#CREATING WIDGETS
dbutils.widgets.text("JOB_NAME","JOB_CIBIL GOOD_BAD_WORKING")
dbutils.widgets.text("COLUMNS_LIST", "ucic_id, score")
#FETCHING VALUES IN WIDGETS AND PUT THIS VALUES IN A VARIABLE 
JOB_NAME = dbutils.widgets.get("JOB_NAME") 
COLUMNS_LIST = dbutils.widgets.get("COLUMNS_LIST")
# then we can use it in notebook as variable 

dbutils.widgets.dropdown("state", "CA", ["CA", "IL", "MI", "NY", "OR", "VA"])

#dbutils.widgets.remove("state")

dbutils.widgets.removeAll()

#Use Databricks widgets with %run:
%run /path/to/notebook $X="10" $Y="1"

######################################################################################################################################################################
1)# Import SparkSession
from pyspark.sql import SparkSession

# Create SparkSession 
spark = SparkSession.builder.master("local[1]") .appName("SparkByExamples.com").getOrCreate() 

#from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('hello').getOrCreate()
df=spark.read.csv(r'C:\Users\sai\Downloads\carprices.csv', header=True)
      
 ####################################################################### RDD  ################################################################################
 
import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
sparkContext=spark.sparkContext

 ## Create RDD from parallelize    
dataList = [("Java", 20000), ("Python", 100000), ("Scala", 3000)]
rdd=spark.sparkContext.parallelize(dataList)

##create empty RDD by using sparkContext.parallelize
emptyRDD = sparkContext.emptyRDD()
emptyRDD2 = rdd=sparkContext.parallelize([])
print("is Empty RDD : "+str(emptyRDD2.isEmpty()))

##Create RDD from external Data source
rdd2 = spark.sparkContext.textFile("/path/test.txt")


##RDD Operations
On PySpark RDD, you can perform two kinds of operations.

RDD transformations – Transformations are lazy operations. When you run a transformation(for example update), instead of updating a current RDD, these operations return another RDD.

RDD actions – operations that trigger computation and return RDD values to the driver.

RDD Transformations
Transformations on Spark RDD returns another RDD and transformations are lazy meaning they don’t execute until you call an action on RDD. Some transformations on RDD’s are flatMap(), map(), reduceByKey(), filter(), sortByKey() and return new RDD instead of updating the current.

RDD Actions
RDD Action operation returns the values from an RDD to a driver node. In other words, any RDD function that returns non RDD[T] is considered as an action. 

Some actions on RDDs are count(), collect(), first(), max(), reduce() and more.

#############################################################create dataframe###########################################################################




5) create dataframe

data = [('James','','Smith','1991-04-01','M',3000),
  ('Michael','Rose','','2000-05-19','M',4000),
  ('Robert','','Williams','1978-09-05','M',4000),
  ('Maria','Anne','Jones','1967-12-01','F',4000),
  ('Jen','Mary','Brown','1980-02-17','F',-1)
]

columns = ["firstname","middlename","lastname","dob","gender","salary"]
df = spark.createDataFrame(data=data, schema = columns)

#SPARKSESSION                         RDD                        DATAFRAME
createDataFrame(rdd)	              toDF()	                 toDF(*cols)
createDataFrame(dataList)	                                   toDF(*cols)	
createDataFrame(rowData,columns)		
createDataFrame(dataList,schema)








################################################### Read files ######################################################################### 

#### Read csv file

df = spark.read.csv("/tmp/resources/zipcodes.csv")
df.printSchema()

#write csv file :

df.write.format("csv").save("/tmp/spark_output/datacsv")


# Read txt file:

df2 = spark.read.text("/src/resources/file.txt")

df1.write.text("output")
// You can specify the compression format using the 'compression' option.
df1.write.option("compression", "gzip").text("output_compressed")



### Read and write  json file

df2 = spark.read.json("/src/resources/file.json")
df2.write.json("/tmp/spark_output/zipcodes.json")

13)Pyspark Read Parquet file into DataFrame

parDF=spark.read.parquet("/tmp/output/people.parquet")

df.write.parquet("/tmp/output/people.parquet")


###Read & Write Avro files using Spark DataFrame

DF= spark.read.format("avro").load("person.avro")

Writing Avro Partition Data

val data = Seq(("James ","","Smith",2018,1,"M",3000),
      ("Michael ","Rose","",2010,3,"M",4000),
      ("Robert ","","Williams",2010,3,"M",4000),
      ("Maria ","Anne","Jones",2005,5,"F",4000),
      ("Jen","Mary","Brown",2010,7,"",-1)
    )

val columns = Seq("firstname", "middlename", "lastname", "dob_year",
 "dob_month", "gender", "salary")
import spark.sqlContext.implicits._
val df = data.toDF(columns:_*)

df.write.partitionBy("dob_year","dob_month")
        .format("avro").save("person_partition.avro")


val data = Seq(("James ","","Smith",2018,1,"M",3000),
      ("Michael ","Rose","",2010,3,"M",4000),
      ("Robert ","","Williams",2010,3,"M",4000),
      ("Maria ","Anne","Jones",2005,5,"F",4000),
      ("Jen","Mary","Brown",2010,7,"",-1)
    )

val columns = Seq("firstname", "middlename", "lastname", "dob_year",
 "dob_month", "gender", "salary")
import spark.sqlContext.implicits._
val df = data.toDF(columns:_*)

# Reading avro partition data

df.write.partitionBy("dob_year","dob_month")
        .format("avro").save("person_partition.avro")
        
        
 # Read ORC file
 df.write.orc('zoo.orc')
spark.read.orc('zoo.orc').show()  

#Read xlsx file 
first  install library in the cluster : Maven> (com.crealytics:spark-excel_2.11:0.12.2)>
Dataframe = spark.read.format(“com.crealytics.spark.excel”)\

.option(“useHeader”, “true”)\

.option(“inferSchema”, “true”)\

.load(“/Location/FileName.xlsx”)

df1.to_excel("output.xlsx")  
df1.to_excel("output.xlsx", sheet_name='Sheet_name_1')  




###################################################### IMP function #############################################################################################
# coalesce: used to decrease partition
Df.coalesce(1). write.mode("overrite'). Parquet (path)

# check partition :  df.rdd.getNumPartition()

#repartition: used to increase or decrease partition 
Df.coalesce(1). write.mode("overrite'). Parquet (path)


# save DataFrame into table:
Df.saveAsTable('table')

#df.createGlobalTempView("people")
df.createOrReplaceGlobalTempView("people")

#

#Create table from csv
df.createOrReplaceTempView("PERSON_DATA")
df2 = spark.sql("SELECT * from PERSON_DATA")

df.head(10)
df.tail()
df.show()
df.columns
df.topandas()
df.printschema
df.describe()
df.select("a", "b", "c").describe().show()
df.collect()

# greatest ():max value of  row /it will create new column with Greatest col name
from pyspark.sql.functions import greatest
greatDF= df.withColumn("Greatest", greatest(" c1", "c2","c3","c4"))

#least():min value of row / will create new col 
from pyspark.sql.functions import least
aleastOf=df.withColumn("Least", least Subject 1", "Subject 2", "Subject_3","Subject_4","Subject 5"))

#df.alias: rename dataframe
from pyspark.sql.functions import *
df_as1 = df.alias("df_as1")
df_as2 = df.alias("df_as2")
joined_df = df_as1.join(df_as2, col("df_as1.name") == col("df_as2.name"), 'inner')
joined_df.select("df_as1.name", "df_as2.name", "df_as2.age")                 .sort(desc("df_as1.name")).collect()
[Row(name='Bob', name='Bob', age=5), Row(name='Alice', name='Alice', age=2)]

#df.createGlobalTempView("people") : select * from global_temp.tempname


#df.describe(['age']).show()
df.describe(['age', 'weight', 'height'])

#df.distinct().count() 

#df.drop('age').collect()
[Row(name='Alice'), Row(name='Bob')]

#df.dropDuplicates().show()
df.dropDuplicates(['name', 'height']).show()

#Returns a new DataFrame omitting rows with null values. DataFrame.dropna() and
df4.na.drop().show()

#df.dtypes

#df.explain(). :for debugging purpose

# fill values on the null place
df4.na.fill(50).show()
df4.na.fill({'age': 50, 'name': 'unknown'}).show()

#df.first() : returns first row

#df1.intersectAll(df2).sort("C1", "C2").show(): returns all row 

#4.na.replace(['Alice', 'Bob'], ['A', 'B'], 'name').show(): 

#df.schema : returns schema of dataframe
StructType([StructField('age', IntegerType(), True),
            StructField('name', StringType

#df.toJSON().first() : convert to json 

#df.c1.cast() :
df.select(df.age.cast("string").alias('ages')).collect()
[Row(ages='2'), Row(ages='5')]
df.select(df.age.cast(StringType()).alias('ages')).collect()
[Row(ages='2'), Row(ages='5')]

#coalesce():
cDf.select(coalesce(cDf["a"], cDf["b"])).show()

# DataFrame.cache() 

#Display full column contents
df.show(truncate=False)

#+-----+-----------------------------------------------------------------------------+
#|Seqno|Quote                                                                        |
#+-----+-----------------------------------------------------------------------------+
#|1    |Be the change that you wish to see in the world                              |
#|2    |Everyone thinks of changing the world, but no one thinks of changing himself.|
#|3    |The purpose of our lives is to be happy.                                     |
#|4    |Be cool.                                                                     |
#+-----+-----------------------------------------------------------------------------+


## Create a Row Object

from pyspark.sql import Row
row=Row("James",40)
print(row[0] +","+str(row[1]))
>>James,40

 ##Select Single & Multiple Columns From PySpark
 
df.select("firstname","lastname").show()
df.select(df.firstname,df.lastname).show()
df.select(df["firstname"],df["lastname"]).show()

#By using col() function
from pyspark.sql.functions import col
df.select(col("firstname"),col("lastname")).show()

#Select columns by regular expression
df.select(df.colRegex("`^.*name*`")).show()
   
   
# Select All columns from List
df.select(*columns).show()

# Select All columns
df.select([col for col in df.columns]).show()
df.select("*").show()


#Selects first 3 columns and top 3 rows
df.select(df.columns[:3]).show(3)

#Selects columns 2 to 4  and top 3 rows
df.select(df.columns[2:4]).show(3)

#select col 1-3 and top 4 rows
df.select(df.columns[2:4]).limit(4)

#PySpark Collect() – Retrieve data from DataFrame

import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()

dept = [("Finance",10), \
    ("Marketing",20), \
    ("Sales",30), \
    ("IT",40) \
  ]
deptColumns = ["dept_name","dept_id"]
deptDF = spark.createDataFrame(data=dept, schema = deptColumns)
deptDF.show(truncate=False)

dataCollect = deptDF.collect()
print(dataCollect)

##Change DataType using PySpark withColumn()
df.withColumn("salary",col("salary").cast("Integer")).show()

#Update The Value of an Existing Column
df.withColumn("salary",col("salary")*100).show()

#Create a Column from an Existing
df.withColumn("CopiedColumn",col("salary")* -1).show()

#Rename Column Name
df.withColumnRenamed("gender","sex") \
  .show(truncate=False)
  
 #with col renamed
 df.withColumnRenamed("dob","DateOfBirth").printSchema()
 
 ## DataFrame filter() with Column Condition
 # Using equals condition
df.filter(df.state == "OH").show(truncate=False)

+----------------------+------------------+-----+------+
|name                  |languages         |state|gender|
+----------------------+------------------+-----+------+
|[James, , Smith]      |[Java, Scala, C++]|OH   |M     |
|[Julia, , Williams]   |[CSharp, VB]      |OH   |F     |
|[Mike, Mary, Williams]|[Python, VB]      |OH   |M     |
+----------------------+------------------+-----+------+

# not equals condition
df.filter(df.state != "OH") \
    .show(truncate=False) 
df.filter(~(df.state == "OH")) \
    .show(truncate=False)
    
    #PySpark Filter with Multiple Conditions
    
//Filter multiple condition
df.filter( (df.state  == "OH") & (df.gender  == "M") ) \
    .show(truncate=False)

# when():
from pyspark.sql.functions import when 
df2= df.withColumn("New status", when(df.Mark>5, "Pass").when(df.Mark <58,"Fai1").otherwise("Absentee"))

from pyspark.sql.functions import when
df4= df.withColumn("Crade", when((df.Mark >*) & (df.Attendance 30), "Distinction") .when((df. Mark >50) & (df.Attendance >50), "Good")
otherwise("Average"))


#Case:
from pyspark.sql.functions import expr
df3=df.withColumn("new,status", expr("CASE WHEN Mark > 50 THEN "Pass" + when mark>50 THEN "Fail" + ELSE "Absentee" ENd))
  
##Get Distinct Rows (By Comparing All Columns)
distinctDF = df.distinct()
print("Distinct count: "+str(distinctDF.count()))
distinctDF.show(truncate=False)

#PySpark Distinct of Selected Multiple Columns
dropDisDF = df.dropDuplicates(["department","salary"])
print("Distinct count of department & salary : "+str(dropDisDF.count()))
dropDisDF.show(truncate=False)

##DataFrame sorting using the sort() function
df.sort("department","state").show(truncate=False)
df.sort(col("department"),col("state")).show(truncate=False)

#DataFrame sorting using orderBy() function
df.orderBy("department","state").show(truncate=False)
df.orderBy(col("department"),col("state")).show(truncate=False)

#Sort by Ascending (ASC)
df.sort(df.department.asc(),df.state.asc()).show(truncate=False)
df.sort(col("department").asc(),col("state").asc()).show(truncate=False)
df.orderBy(col("department").asc(),col("state").asc()).show(truncate=False)

#Sort by Descending (DESC)
df.sort(df.department.asc(),df.state.desc()).show(truncate=False)
df.sort(col("department").asc(),col("state").desc()).show(truncate=False)
df.orderBy(col("department").asc(),col("state").desc()).show(truncate=False)

##PySpark groupBy and aggregate on DataFrame columns
df.groupBy("department").count()
df.groupBy("department").min("salary")
df.groupBy("department").max("salary")
df.groupBy("department").avg( "salary")
df.groupBy("department").mean( "salary") 

#PySpark groupBy and aggregate on multiple columns
//GroupBy on multiple columns
df.groupBy("department","state") \
    .sum("salary","bonus") \
    .show(false)

#Running more aggregates at a time
df.groupBy("department") \
    .agg(sum("salary").alias("sum_salary"), \
         avg("salary").alias("avg_salary"), \
         sum("bonus").alias("sum_bonus"), \
         max("bonus").alias("max_bonus") \
     ) \
    .show(truncate=False)

#Using filter on aggregate data
df.groupBy("department") \
    .agg(sum("salary").alias("sum_salary"), \
      avg("salary").alias("avg_salary"), \
      sum("bonus").alias("sum_bonus"), \
      max("bonus").alias("max_bonus")) \
    .where(col("sum_bonus") >= 50000) \
    .show(truncate=False)

### Join
#PySpark Inner Join DataFrame
empDF.join(deptDF,empDF.emp_dept_id ==  deptDF.dept_id,"inner") \
     .show(truncate=False)

#PySpark Full Outer Join
empDF.join(deptDF,empDF.emp_dept_id ==  deptDF.dept_id,"outer") \
    .show(truncate=False)
empDF.join(deptDF,empDF.emp_dept_id ==  deptDF.dept_id,"full") \
    .show(truncate=False)
empDF.join(deptDF,empDF.emp_dept_id ==  deptDF.dept_id,"fullouter") \
    .show(truncate=False)

#PySpark Left Outer Join
 empDF.join(deptDF,empDF.emp_dept_id ==  deptDF.dept_id,"left")
    .show(truncate=False)
  empDF.join(deptDF,empDF.emp_dept_id ==  deptDF.dept_id,"leftouter")
    .show(truncate=False)

#Right Outer Join
empDF.join(deptDF,empDF.emp_dept_id ==  deptDF.dept_id,"right") \
   .show(truncate=False)
empDF.join(deptDF,empDF.emp_dept_id ==  deptDF.dept_id,"rightouter") \
   .show(truncate=False)

#Using SQL Expression
empDF.createOrReplaceTempView("EMP")
deptDF.createOrReplaceTempView("DEPT")

joinDF = spark.sql("select * from EMP e, DEPT d where e.emp_dept_id == d.dept_id") \
  .show(truncate=False)

joinDF2 = spark.sql("select * from EMP e INNER JOIN DEPT d ON e.emp_dept_id == d.dept_id") \
  .show(truncate=False)

## union or unionall
#Merge two or more DataFrames using union
unionDF = df.union(df2)
unionDF.show(truncate=False)

#Merge DataFrames using unionAll
unionAllDF = df.unionAll(df2)
unionAllDF.show(truncate=False)

#Merge without Duplicates
disDF = df.union(df2).distinct()
disDF.show(truncate=False)

#Spark Merge Two DataFrames with Different Columns or Schema
//Scala
merged_df = df1.unionByName(df2, true)

#PySpark
merged_df = df1.unionByName(df2, allowMissingColumns=True)

###Create a Python Function
def convertCase(str):
    resStr=""
    arr = str.split(" ")
    for x in arr:
       resStr= resStr + x[0:1].upper() + x[1:len(x)] + " "
    return resStr 

#Converting function to UDF """
convertUDF = udf(lambda z: convertCase(z),StringType())

###Using foreach() to Loop Through Rows in DataFrame
# Foreach example
def f(x): print(x)
df.foreach(f)

# Another example
df.foreach(lambda x: 
    print("Data ==>"+x["firstname"]+","+x["lastname"]+","+x["gender"]+","+str(x["salary"]*2))
    ) 

#Using pandas() to Iterate
# Using pandas
import pandas as pd
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
pandasDF = df.toPandas()
for index, row in pandasDF.iterrows():
    print(row['firstname'], row['gender'])

#Collect Data As List and Loop Through
# Collect the data to Python List
dataCollect = df.collect()
for row in dataCollect:
    print(row['firstname'] + "," +row['lastname'])

#Using toLocalIterator()
dataCollect=df.rdd.toLocalIterator()
for row in dataCollect:
    print(row['firstname'] + "," +row['lastname'])

###PySpark fillna() & fill() – Replace NULL/None Values
#Replace 0 for null for all integer columns
df.na.fill(value=0).show()

#Replace 0 for null on only population column 
df.na.fill(value=0,subset=["population"]).show()

#PySpark Replace Null/None Value with Empty String
df.na.fill("").show(false)

###Pivot PySpark DataFrame PySpark
pivotDF = df.groupBy("Product").pivot("Country").sum("Amount")
pivotDF.printSchema()
pivotDF.show(truncate=False)

#Unpivot PySpark DataFrame
from pyspark.sql.functions import expr
unpivotExpr = "stack(3, 'Canada', Canada, 'China', China, 'Mexico', Mexico) as (Country,Total)"
unPivotDF = pivotDF.select("Product", expr(unpivotExpr)) \
    .where("Total is not null")
unPivotDF.show(truncate=False)
unPivotDF.show()

###PySpark partitionBy()
PySpark partitionBy() is a function of pyspark
.sql.DataFrameWriter class which is used to 
partition based on column values while writing
DataFrame to Disk/File system.


#partitionBy()
df.write.option("header",True) \
        .partitionBy("state") \
        .mode("overwrite") \
        .csv("/tmp/zipcodes-state")


#partitionBy() multiple columns
df.write.option("header",True) \
        .partitionBy("state","city") \
        .mode("overwrite") \
        .csv("/tmp/zipcodes-state")

#Data Skew – Control Number of Records per Partition File
#partitionBy() control number of partitions
df.write.option("header",True) \
        .option("maxRecordsPerFile", 2) \
        .partitionBy("state") \
        .mode("overwrite") \
        .csv("/tmp/zipcodes-state")


###PySpark Aggregate Functions with Examples
#approx_count_distinct Aggregate Function
In PySpark approx_count_distinct() function
returns the count of distinct items in a
group.

//approx_count_distinct()
print("approx_count_distinct: " + \
      str(df.select(approx_count_distinct("salary")).collect()[0][0]))

//Prints approx_count_distinct: 6

#avg (average) Aggregate Function
print("avg: " + str(df.select(avg("salary")).collect()[0][0]))

//Prints avg: 3400.0

#collect_list Aggregate Function
collect_list() function returns all values from an input column with duplicates

df.select(collect_list("salary")).show(truncate=False)

+------------------------------------------------------------+
|collect_list(salary)                                        |
+------------------------------------------------------------+
|[3000, 4600, 4100, 3000, 3000, 3300, 3900, 3000, 2000, 4100]|
+------------------------------------------------------------+

#countDistinct Aggregate Function
countDistinct() function returns the number of distinct elements in a columns
df2 = df.select(countDistinct("department", "salary"))
df2.show(truncate=False)
print("Distinct Count of Department & Salary: "+str(df2.collect()[0][0]))

#count function
count() function returns number of elements in a column

print("count: "+str(df.select(count("salary")).collect()[0]))

Prints county: 10


#first function
first() function returns the first element in a column when ignoreNulls is set to true, it returns the first non-null element

//first
df.select(first("salary")).show(truncate=False)

+--------------------+
|first(salary, false)|
+--------------------+
|3000                |
+--------------------+


#last function
last() function returns the last element in a column. when ignoreNulls is set to true, it returns the last non-null element

//last
df.select(last("salary")).show(truncate=False)

+-------------------+
|last(salary, false)|
+-------------------+
|4100               |
+-------------------+


#kurtosis function
kurtosis() function returns the kurtosis of the values in a group

df.select(kurtosis("salary")).show(truncate=False)

+-------------------+
|kurtosis(salary)   |
+-------------------+
|-0.6467803030303032|
+-------------------+

#max function

df.select(max("salary")).show(truncate=False)

+-----------+
|max(salary)|
+-----------+
|4600       |
+-----------+


#min function
df.select(min("salary")).show(truncate=False)

+-----------+
|min(salary)|
+-----------+
|2000       |
+-----------+

#mean function

df.select(mean("salary")).show(truncate=False)

+-----------+
|avg(salary)|
+-----------+
|3400.0     |
+-----------+


#skewness function
skewness() function returns the skewness of the values in a group

df.select(skewness("salary")).show(truncate=False)

+--------------------+
|skewness(salary)    |
+--------------------+
|-0.12041791181069571|
+--------------------+


#stddev(), stddev_samp() and stddev_pop()
stddev() alias for stddev_samp.

stddev_samp() function returns the sample standard deviation of values in a column.

stddev_pop() function returns the population standard deviation of the values in a column


df.select(stddev("salary"), stddev_samp("salary"), \
    stddev_pop("salary")).show(truncate=False)

+-------------------+-------------------+------------------+
|stddev_samp(salary)|stddev_samp(salary)|stddev_pop(salary)|
+-------------------+-------------------+------------------+
|765.9416862050705  |765.9416862050705  |726.636084983398  |
+-------------------+-------------------+------------------+


#sum function

df.select(sum("salary")).show(truncate=False)

+-----------+
|sum(salary)|
+-----------+
|34000      |
+-----------+

#sumDistinct function
sumDistinct() function returns the sum of all distinct values in a column

df.select(sumDistinct("salary")).show(truncate=False)

+--------------------+
|sum(DISTINCT salary)|
+--------------------+
|20900               |
+--------------------+


#variance(), var_samp(), var_pop()
variance() alias for var_samp

var_samp() function returns the unbiased variance of the values in a column.

var_pop() function returns the population variance of the values in a column


df.select(variance("salary"),var_samp("salary"),var_pop("salary")) \
  .show(truncate=False)

+-----------------+-----------------+---------------+
|var_samp(salary) |var_samp(salary) |var_pop(salary)|
+-----------------+-----------------+---------------+
|586666.6666666666|586666.6666666666|528000.0       |
+-----------------+-----------------+---------------+

### intersect : combine df on the basis of rows and it removes duplicate 
df_inter=df_summerfruits.intersect(df_fruits)
df_inter.show()

### intersectAll:
Intersect all of the dataframe in pyspark is similar to intersect function but the only difference is it will not remove the duplicate rows of the resultant dataframe

df_summerfruits.intersectAll(df_fruits).show()

###EXCEPT:
EXCEPT and EXCEPT ALL return the rows that are found in one relation but not the other. EXCEPT (alternatively, EXCEPT DISTINCT) takes only distinct rows while EXCEPT ALL does not remove duplicates from the result rows. Note that MINUS is an alias for EXCEPT
DF1.except(DF2)

Except : removes duplicate
Exceptall: keeps duplicate 

df1 = spark.createDataFrame(
        [("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3), ("c", 4)], ["C1", "C2"])
df2 = spark.createDataFrame([("a", 1), ("b", 3)], ["C1", "C2"])

df1.exceptAll(df2).show()
+---+---+
| C1| C2|
+---+---+
|  a|  1|
|  a|  1|
|  a|  2|
|  c|  4|

+---+---+

#all colum n at the same time:
for col in df.columns:
      df=df.withColumn( col ,trim (col))
############################################ String functions ######################################
# concat_ws(): 
concatenation string with separator
df = spark.createDataFrame([('abcd','123')], ['s', 'd'])
df.select(concat_ws('-', df.s, df.d).alias('s')).collect()
[Row(s='abcd-123')]

#format_string(): 
it combines two col
df = spark.createDataFrame([(5, "hello")], ['a', 'b'])
df.select(format_string('%d %s', df.a, df.b).alias('v')).collect()
[Row(v='5 hello')]

#initcap():
Translate the first letter of each word to upper case in the sentence
spark.createDataFrame([('ab cd',)], ['a']).select(initcap("a").alias('v')).collect()
[Row(v='Ab Cd')]

#instr(): 
Locate the position of the first occurrence of substr column in the given string
df = spark.createDataFrame([('abcd',)], ['s',])
df.select(instr(df.s, 'b').alias('s')).collect()
[Row(s=2)]

#length():
spark.createDataFrame([('ABC ',)], ['a']).select(length('a').alias('length')).collect()
[Row(length=4)]

#locate: substring in String 
df = spark.createDataFrame([('abcd',)], ['s',])
df.select(locate('b', df.s, 1).alias('s')).collect()
[Row(s=2)]

# regexp_replace():
df = spark.createDataFrame([('100-200',)], ['str'])
df.select(regexp_replace('str', r'(\d+)', '--').alias('d')).collect()
[Row(d='-----')]

#repeat():
df = spark.createDataFrame([('ab',)], ['s',])
df.select(repeat(df.s, 3).alias('s')).collect()
[Row(s='ababab')]

# split ():
df = spark.createDataFrame([('oneAtwoBthreeC',)], ['s',])
df.select(split(df.s, '[ABC]', 2).alias('s')).collect()
[Row(s=['one', 'twoBthreeC'])]
df.select(split(df.s, '[ABC]', -1).alias('s')).collect()
[Row(s=['one', 'two', 'three', ''])]

#substring():Substring starts at pos and is of length len when str is String type or returns the slice of byte array that starts at pos in byte and is of length len when str is Binary type
df = spark.createDataFrame([('abcd',)], ['s',])
df.select(substring(df.s, 1, 2).alias('s')).collect()
[Row(s='ab')]

#upper()
#lower()
#trim():

#translate():
spark.createDataFrame([('translate',)], ['a']).select(translate('a', "rnlt", "123") \
    .alias('r')).collect()
[Row(r='1a2s3ae')]

#sentences():
df = spark.createDataFrame([["This is an example sentence."]], ["string"])
df.select(sentences(df.string, lit("en"), lit("US"))).show(truncate=False)
+-----------------------------------+
|sentences(string, en, US)          |
+-----------------------------------+
|[[This, is, an, example, sentence]]|
+-----------------------------------+

#

######################################### Datatype & Data casting #################################################################################################
from pyspark.sql.types import *
df2.select(to_date(df2.BirthDate)).alias('Birth_Date').printSchema()

from pyspark.sql.functions import to_timestamp
 df = spark.createDataFrame([('1997-02-28 10:30:00',)], ['t'])
df.select(to_timestamp(df.t, 'yyyy-MM-dd HH:mm:ss').alias('dt')).collect()
>>[Row(dt=datetime.datetime(1997, 2, 28, 10, 30)

Data type                                 	Value type in Python	                           API to access or create a data type
1]IntegerType                                    	int or long	                                                 IntegerType()
2]FloatType	                                     float Note: Numbers will be converted to 4-byte 
                                               single-precision floating point numbers at runtime.	               FloatType()
                                               
3]DecimalType	                                           decimal.Decimal                                             DecimalType()    
4]StringType	                                               string	                                                 StringType()  
5]BinaryType	                                             bytearray	                                            BinaryType()    
6]BooleanType                                                  	bool	                                                  BooleanType()
7]TimestampType	                                         datetime.datetime                                        	TimestampType()
8]DateType                                             	datetime.date	                                            DateType()
9]ArrayType                                          	list, tuple, or array
10]MapType                                                   	dict
11]StructType	                                        list or tuple


# Int conversion:

                                               
                                               

  








################################################################Window Functions##################################################################################

PySpark Window functions are used to calculate results such as the rank, row number e.t.c over a range of input rows. In this article, I’ve explained the concept of window functions, syntax, and finally how to use them with PySpark SQL and PySpark DataFrame API. These come in handy when we need to make aggregate operations in a specific window frame on DataFrame columns

###row_number Window Function
row_number() window function is used to give the sequential row number starting from 1 to the result of each window partition

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
windowSpec  = Window.partitionBy("department").orderBy("salary")

df.withColumn("row_number",row_number().over(windowSpec)) \
    .show(truncate=False)

>>>
+-------------+----------+------+----------+
|employee_name|department|salary|row_number|
+-------------+----------+------+----------+
|James        |Sales     |3000  |1         |
|James        |Sales     |3000  |2         |
|Robert       |Sales     |4100  |3         |
|Saif         |Sales     |4100  |4         |
|Michael      |Sales     |4600  |5         |
|Maria        |Finance   |3000  |1         |
|Scott        |Finance   |3300  |2         |
|Jen          |Finance   |3900  |3         |
|Kumar        |Marketing |2000  |1         |
|Jeff         |Marketing |3000  |2         |
+-------------+----------+------+----------+

###rank Window Function
rank() window function is used to provide a rank to the result within a window partition. This function leaves gaps in rank when there


"""rank"""
from pyspark.sql.functions import rank
df.withColumn("rank",rank().over(windowSpec)) \
    .show()

>>>
+-------------+----------+------+----+
|employee_name|department|salary|rank|
+-------------+----------+------+----+
|        James|     Sales|  3000|   1|
|        James|     Sales|  3000|   1|
|       Robert|     Sales|  4100|   3|
|         Saif|     Sales|  4100|   3|
|      Michael|     Sales|  4600|   5|
|        Maria|   Finance|  3000|   1|
|        Scott|   Finance|  3300|   2|
|          Jen|   Finance|  3900|   3|
|        Kumar| Marketing|  2000|   1|
|         Jeff| Marketing|  3000|   2|
+-------------+----------+------+----+


###dense_rank Window Function
dense_rank() window function is used to get the result with rank of rows within a window partition without any gaps. This is similar to rank() function difference being rank function leaves gaps in rank when there are ties


"""dens_rank"""
from pyspark.sql.functions import dense_rank
df.withColumn("dense_rank",dense_rank().over(windowSpec)) \
    .show()

>>>
+-------------+----------+------+----------+
|employee_name|department|salary|dense_rank|
+-------------+----------+------+----------+
|        James|     Sales|  3000|         1|
|        James|     Sales|  3000|         1|
|       Robert|     Sales|  4100|         2|
|         Saif|     Sales|  4100|         2|
|      Michael|     Sales|  4600|         3|
|        Maria|   Finance|  3000|         1|
|        Scott|   Finance|  3300|         2|
|          Jen|   Finance|  3900|         3|
|        Kumar| Marketing|  2000|         1|
|         Jeff| Marketing|  3000|         2|
+-------------+----------+------+----------+

###percent_rank Window Function

""" percent_rank """
from pyspark.sql.functions import percent_rank
df.withColumn("percent_rank",percent_rank().over(windowSpec)) \
    .show()

>>>
+-------------+----------+------+------------+
|employee_name|department|salary|percent_rank|
+-------------+----------+------+------------+
|        James|     Sales|  3000|         0.0|
|        James|     Sales|  3000|         0.0|
|       Robert|     Sales|  4100|         0.5|
|         Saif|     Sales|  4100|         0.5|
|      Michael|     Sales|  4600|         1.0|
|        Maria|   Finance|  3000|         0.0|
|        Scott|   Finance|  3300|         0.5|
|          Jen|   Finance|  3900|         1.0|
|        Kumar| Marketing|  2000|         0.0|
|         Jeff| Marketing|  3000|         1.0|
+-------------+----------+------+------------+


###ntile Window Function
ntile() window function returns the relative rank of result rows within a window partition. In below example we have used 2 as an argument to ntile hence it returns ranking between 2 values (1 and 2)


"""ntile"""
from pyspark.sql.functions import ntile
df.withColumn("ntile",ntile(2).over(windowSpec)) \
    .show()

>>>
+-------------+----------+------+-----+
|employee_name|department|salary|ntile|
+-------------+----------+------+-----+
|        James|     Sales|  3000|    1|
|        James|     Sales|  3000|    1|
|       Robert|     Sales|  4100|    1|
|         Saif|     Sales|  4100|    2|
|      Michael|     Sales|  4600|    2|
|        Maria|   Finance|  3000|    1|
|        Scott|   Finance|  3300|    1|
|          Jen|   Finance|  3900|    2|
|        Kumar| Marketing|  2000|    1|
|         Jeff| Marketing|  3000|    2|
+-------------+----------+------+-----+


#### PySpark Window Analytic functions
###cume_dist Window Function
cume_dist() window function is used to get the cumulative distribution of values within a window partition.

This is the same as the DENSE_RANK function in SQL


""" cume_dist """
from pyspark.sql.functions import cume_dist    
df.withColumn("cume_dist",cume_dist().over(windowSpec)) \
   .show()

>>>
+-------------+----------+------+------------------+
|employee_name|department|salary|         cume_dist|
+-------------+----------+------+------------------+
|        James|     Sales|  3000|               0.4|
|        James|     Sales|  3000|               0.4|
|       Robert|     Sales|  4100|               0.8|
|         Saif|     Sales|  4100|               0.8|
|      Michael|     Sales|  4600|               1.0|
|        Maria|   Finance|  3000|0.3333333333333333|
|        Scott|   Finance|  3300|0.6666666666666666|
|          Jen|   Finance|  3900|               1.0|
|        Kumar| Marketing|  2000|               0.5|
|         Jeff| Marketing|  3000|               1.0|
+-------------+----------+------+------------------+


### lag Window Function
This is the same as the LAG function in SQL

"""lag"""
from pyspark.sql.functions import lag    
df.withColumn("lag",lag("salary",2).over(windowSpec)) \
      .show()

>>>
+-------------+----------+------+----+
|employee_name|department|salary| lag|
+-------------+----------+------+----+
|        James|     Sales|  3000|null|
|        James|     Sales|  3000|null|
|       Robert|     Sales|  4100|3000|
|         Saif|     Sales|  4100|3000|
|      Michael|     Sales|  4600|4100|
|        Maria|   Finance|  3000|null|
|        Scott|   Finance|  3300|null|
|          Jen|   Finance|  3900|3000|
|        Kumar| Marketing|  2000|null|
|         Jeff| Marketing|  3000|null|
+-------------+----------+------+----+


###lead Window Function
This is the same as the LEAD function in SQL.

 """lead"""
from pyspark.sql.functions import lead    
df.withColumn("lead",lead("salary",2).over(windowSpec)) \
    .show()

>>>
+-------------+----------+------+----+
|employee_name|department|salary|lead|
+-------------+----------+------+----+
|        James|     Sales|  3000|4100|
|        James|     Sales|  3000|4100|
|       Robert|     Sales|  4100|4600|
|         Saif|     Sales|  4100|null|
|      Michael|     Sales|  4600|null|
|        Maria|   Finance|  3000|3900|
|        Scott|   Finance|  3300|null|
|          Jen|   Finance|  3900|null|
|        Kumar| Marketing|  2000|null|
|         Jeff| Marketing|  3000|null|
+-------------+----------+------+----+


#### PySpark Window Aggregate Functions

windowSpecAgg  = Window.partitionBy("department")
from pyspark.sql.functions import col,avg,sum,min,max,row_number 
df.withColumn("row",row_number().over(windowSpec)) \
  .withColumn("avg", avg(col("salary")).over(windowSpecAgg)) \
  .withColumn("sum", sum(col("salary")).over(windowSpecAgg)) \
  .withColumn("min", min(col("salary")).over(windowSpecAgg)) \
  .withColumn("max", max(col("salary")).over(windowSpecAgg)) \
  .where(col("row")==1).select("department","avg","sum","min","max") \
  .show()


 >>>
+----------+------+-----+----+----+
|department|   avg|  sum| min| max|
+----------+------+-----+----+----+
|     Sales|3760.0|18800|3000|4600|
|   Finance|3400.0|10200|3000|3900|
| Marketing|2500.0| 5000|2000|3000|
+----------+------+-----+----+----+



############################################# Access adls gen2 and blob using sas in azure databricks #####################################################################################

#Access ADLS Gen2 with SAS Token:
Step 1:
spark.conf.set("fs.azure.account.auth.type.<storage account>.dfs.core.windows.net", "SAS")

spark.conf.set("fs.azure.sas.token.provider.type.<storage account>.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas. FixedSASToken Provider")

spark.conf.set("fs.azure.sas.fixed.token.<storage account>.dfs.core.windows.net", "<token>")
 
Step 2:
df = spark.read.csv("abfs://samplecontainer@<container name>.dfs.core.windows.net/data/Employees.csv",header=True)

display(df)


################## SQL Server function #############################################################################
Different categories of window functions:
-Aggregate functions-:   AVG, SUM, COUNT, MIN, MAX etc..
-Ranking functions-:     RANK, DENSE_RANK, ROW_NUMBER etc..
-Analytic functions-:    LEAD, LAG, FIRST_VALUE, LAST_VALUE etc...


#Rank function:























































