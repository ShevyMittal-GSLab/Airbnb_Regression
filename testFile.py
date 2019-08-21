
from pyspark.sql import SparkSession

warehouse_location = "/apps/spark/warehouse"
spark = SparkSession.builder.config("spark.sql.warehouse.dir", warehouse_location).getOrCreate()
df = spark.sql('show databases').toPandas()
print(df)
df2 =  spark.sql('show tables').toPandas()
print(df2)