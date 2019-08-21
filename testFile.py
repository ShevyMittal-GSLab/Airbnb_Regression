
from pyspark.sql import SparkSession

warehouse_location = "/apps/spark"
spark = SparkSession.builder.getOrCreate()
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(spark)
print(spark.master)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>2222")
print(spark.sparkContext.getConf().getAll())
df = spark.sql('show databases').toPandas()
print(df)
df2 =  spark.sql('show tables').toPandas()
print(df2)