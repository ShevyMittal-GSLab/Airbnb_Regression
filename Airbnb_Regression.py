

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
 
def airbnb_regression():
	print("STARTED>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
	spark = SparkSession.builder.config('spark.sql.catalogImplementation','hive').getOrCreate()
	print("1>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
	df = spark.sql('select * from knime_datasets.queens').toPandas() 
	print("2>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
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
	height = [5010, 32, 2, 39, 4, 23, 7, 225, 83, 254, 311, 1933, 2333, 5412, 4227]
	bars = ['100', '20', '30', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95']
	y_pos = np.arange(len(bars))
	plt.bar(y_pos, height)
	plt.xticks(y_pos, bars)
	plt.ylabel("count")
	plt.xlabel(target)
	plt.show()
	plt.savefig("plot.png")
	def eval_metrics(actual, pred):
		rmse = np.sqrt(mean_squared_error(actual, pred))
		mae = mean_absolute_error(actual, pred)
		r2 = r2_score(actual, pred)
		return rmse, mae, r2
	alpha = 10
	learning_rate = 0.1
	colsample_bytree = 0.3
	max_depth = 5
	objective = reg:linear
	n_estimators = 10
	subsample = None
	gamma = None
	lambda1 = None

	mlflow.set_tracking_uri("http://10.43.13.1:5000")
	experiment_name = "Airbnb_Regression"
	mlflow.set_experiment(experiment_name)
	with mlflow.start_run():
		#lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
		#lr.fit(train_x, train_y)
		xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)
		#predicted_qualities = lr.predict(test_x)
		xg_reg.fit(train_x, train_y)
		predicted_qualities = xg_reg.predict(test_x)
		(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
		
		print("XGBoost model")
		print("  RMSE: %s" % rmse)
		print("  MAE: %s" % mae)
		print("  R2: %s" % r2)
		
		mlflow.log_param("objective", objective)
		mlflow.log_param("colsample_bytree", colsample_bytree)
		mlflow.log_param("learning_rate", learning_rate)
		mlflow.log_param("max_depth", max_depth)
		mlflow.log_param("alpha", alpha)
		mlflow.log_param("n_estimators", n_estimators)

		mlflow.log_param("Model","XGBoost")
		mlflow.log_metric("rmse", rmse)
		mlflow.log_metric("r2", r2)
		mlflow.log_metric("mae", mae)
		mlflow.log_artifact("plot.png")
		print("Logging Model")
		#mlflow.sklearn.log_model(lr,".")
		mlflow.sklearn.log_model(xg_reg,".")
		print("Model Logged")
		runId = mlflow.active_run().info.run_id
		expId = mlflow.active_run().info.experiment_id
		artifact_uri = mlflow.active_run().info.artifact_uri
	

