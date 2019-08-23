
from pyspark.sql import SparkSession

def testFile():
	import matplotlib.pyplot as plt
	import pandas as pd
	import numpy as np
	from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import ElasticNet
	from pyspark.sql import SparkSession
	import mlflow
	import mlflow.sklearn
	import mlflow.models
	import xgboost as xgb
	spark = SparkSession.builder.config('spark.sql.catalogImplementation','hive').getOrCreate()
	df = spark.sql('select * from knime_datasets.queens').toPandas()
	target = "Review_Scores_Rating5"
	df = df[df.Review_Scores_Rating5.notnull()]
	df = df[df.Number_of_Records.notnull()]
	df = df[df.Number_Of_Reviews.notnull()]
	df = df[df.Review_Scores_Rating12.notnull()]
	df['Review_Scores_Rating5'] = df.Review_Scores_Rating5.astype(float)
	df['Number_of_Records'] = df.Number_of_Records.astype(float)
	df['Number_Of_Reviews'] = df.Number_Of_Reviews.astype(float)
	df['Review_Scores_Rating12'] = df.Review_Scores_Rating12.astype(float)
	df['Review_Scores_Rating5'] = df.Review_Scores_Rating5.astype(float)
	train, test = train_test_split(df)
	train_x = train[['Number_of_Records', 'Number_Of_Reviews', 'Review_Scores_Rating12']]
	test_x = test[['Number_of_Records', 'Number_Of_Reviews', 'Review_Scores_Rating12']]
	train_y = train[["Review_Scores_Rating5"]]
	test_y = test[["Review_Scores_Rating5"]]
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
	print(df)
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>2222")
	
if __name__ == '__main__':
    testFile()