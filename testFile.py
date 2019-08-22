
from pyspark.sql import SparkSession

warehouse_location = "/apps/spark"
spark = SparkSession.builder.config('spark.sql.catalogImplementation','hive').getOrCreate()
df = spark.sql('select * from knime_datasets.queens')
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(df)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>2222")
df = spark.sql('show databases').toPandas()
print(df)
df2 =  spark.sql('show tables').toPandas()
print(df2)