# Databricks notebook source
# MAGIC %fs ls dbfs:/databricks-datasets/bikeSharing

# COMMAND ----------

with open("/dbfs/databricks-datasets/bikeSharing/README.md", "r") as f_read:
    for line in f_read:
        print(line)


# COMMAND ----------

# Load the data 
bikeDF = (spark
          .read
          .option("header", True)
          .option("inferSchema", True) .csv("dbfs:/databricks-datasets/bikeSharing/data-001/hour.csv") .drop("instant", "dteday", "casual", "registered", "holiday", "weekday")
)

# COMMAND ----------

# Train/Test Split
trainBikeDF, testBikeDF = bikeDF.randomSplit([0.7, 0.3])
print(f"Training samples: {trainBikeDF.count()}, Testing samples: {testBikeDF.count()}")

# COMMAND ----------

# Creating a baseline
from pyspark.sql.functions import avg, lit
from pyspark.ml.evaluation import RegressionEvaluator 
from math import sqrt

avgTrainCnt = trainBikeDF.select(avg("cnt")).first()[0]
bikeTestPredictionDF = testBikeDF.withColumn("base_prediction", lit(avgTrainCnt))

evaluator = RegressionEvaluator(predictionCol="base_prediction", labelCol="cnt", metricName="rmse")
error = evaluator.evaluate(bikeTestPredictionDF)
bikeCounts = sqrt(error)

print(f"Root mean squared error (RMSE) on the test set for the baseline model: {error}")


# COMMAND ----------

# Prepare pipeline
from pyspark.ml.feature import VectorAssembler, VectorIndexer 
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

featuresCols = bikeDF.columns
featuresCols.remove('cnt')
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures") 
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)
gbt = GBTRegressor(labelCol="cnt")

pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, gbt])

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol=gbt.getLabelCol(),
    predictionCol=gbt.getPredictionCol()
)


# COMMAND ----------

# Train and evaluate GBT model
pipelineModel = pipeline.fit(trainBikeDF)
predictions = pipelineModel.transform(testBikeDF)
error = evaluator.evaluate(predictions)
print(f"Root mean squared error (RMSE) on the test set for the default GBT model: {error}")

# COMMAND ----------

# Tune Model hyperparameters
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 
from pyspark.ml.evaluation import RegressionEvaluator
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 5])
             .addGrid(gbt.maxIter, [10, 100])
             .build()
)
# .addGrid(gbt.maxDepth, [2, 4, 8, 16])
# .addGrid(gbt.maxIter, [10, 100, 200])

crossValidator = CrossValidator(
    estimator=gbt,
    evaluator=evaluator,
    estimatorParamMaps=paramGrid
)

pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, crossValidator])

# COMMAND ----------

# Train hyperparameter tuned model
pipelineModel = pipeline.fit(trainBikeDF)
predictions = pipelineModel.transform(testBikeDF)
error = evaluator.evaluate(predictions)
print(f"Root mean squared error (RMSE) on the test set for the hyperparameter tuned GBT model: {error}")

# COMMAND ----------



# COMMAND ----------

dbutils.notebook.exit()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Written Answer
# MAGIC Reference: ML Pipelines - Analyzing a bike sharing dataset
# MAGIC https://databricks-prod- cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2854662143668609/208478869198391 (https://databricks-prod- cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2854662143668609/208478869198391
# MAGIC Setup
# MAGIC Data Ingestion and preprocessing
# MAGIC Always check the README prior to loading data. It also helps understand the fields better to make a selection of columns prior on load. From the README we find that we would like to predict the count of bike renatals (cnt column). We can remove any data leakage columns such as 'casual' and 'registered', make an informed selection or first attempt on date related columns and drop any useless columns.
# MAGIC Train Test Split
# MAGIC The training set is the base used to create the model and the test set is used to validate the model on unseen data. Splits are important to avoid overfitting and allowing us check model performance. Generally a 80-20 or 70-30 split is a good place to start prior to any cross validation techniques. We can use the randomSplit method to split our data
# MAGIC Establish a baseline
# MAGIC file:///C:/Users/fxpr/Downloads/Solution Architect Coding Assignment Final.html
# MAGIC 36/39
# MAGIC  8/30/2020
# MAGIC Solution Architect Coding Assignment Final - Databricks
# MAGIC It if often a good idea to start with a baseline model which can be something simple like the average and then tune models to improve upon each other. Since this is a regression problem we can use the RegressionEvaluator class to compare or test dataset with the baseline average. There are many metrics to choose from but mean squared error is a good start for regression problems. A key to understanding metrics is to always remember to transform the metric back to the proper units, in this case total bike rentals in which case we should use root mean squared error (RMSE).
# MAGIC ML Pipeline
# MAGIC Spark's machine learning library MLlib has three main abstractions:
# MAGIC 1. transformer: Implemented through the .transfrom() method, takes a dataframe and returns a dataframe with
# MAGIC columns appended.
# MAGIC 2. estimator Implemented with a .fit() method, takes a dataframe and returns a model
# MAGIC 3. pipeline Implemnted with a .fit() method, combines together transformers and estimators to chain together
# MAGIC algorithms
# MAGIC Featurization
# MAGIC Reference: https://spark.apache.org/docs/latest/ml-features (https://spark.apache.org/docs/latest/ml-features)
# MAGIC In order to do machine learning in a distributed fashion the features need to be assembled into a feature vector with the VectorAssembler class. Since we already did some preprocessing of the data we will use all the columns and remove the target variable cnt.
# MAGIC The VectorIndexer is used to identify columns which should be treated as categorical
# MAGIC Model training stage
# MAGIC file:///C:/Users/fxpr/Downloads/Solution Architect Coding Assignment Final.html
# MAGIC 37/39
# MAGIC 
# MAGIC  8/30/2020
# MAGIC Solution Architect Coding Assignment Final - Databricks
# MAGIC Reference: https://en.wikipedia.org/wiki/Gradient_boosting (https://en.wikipedia.org/wiki/Gradient_boosting) Gradient boosting can be used for both regression and classification problems by means of creating an ensemble of weak decision tree prediction models. Decision tree based models are a good first choice as they require little preprocessing, easy to implement, and performs well both in terms of compute and metrics. XGboost is often a better choice in terms of performance and accuracy. But since XGboost requires aditional cluster configuration it was not implemented here (https://docs.databricks.com/applications/machine-learning/train-model/machine-learning.html#xgboost (https://docs.databricks.com/applications/machine-learning/train-model/machine-learning.html#xgboost)). The pipeline is set up so we have feature processing (vector assembler and vector indexer) and then the GBT regressor. We can now cal fit to produce a model to be evaluated. Comparing the RMSE from the default GBT to the initial baseline we are clearly going in the right direction! GBT operate by minimizing some metric in an iterative fashion. The RMSE measures the the difference between the esimator and the observed values. RMSE is non-negative and a value closer to 0 indicateds a better fit to the data.
# MAGIC Tuning
# MAGIC Tuning the model can encompass many things but mainly it is around: 1. Model selection
# MAGIC 2. Domain knowledge and feature selection
# MAGIC 3. Hyperparameter tuning
# MAGIC 4. Feature engineering 5. Ensemble models
# MAGIC Here we can wrap the GBT regressor within a CrossValidator stage with a parameter grid for hyperparamters. We set the fold to three in a subsequent run to validate the results and generalize the model. Depending on the model there are many different hyperparameters to tune but for decision tree based models the depth and iterations are considered
# MAGIC file:///C:/Users/fxpr/Downloads/Solution Architect Coding Assignment Final.html
# MAGIC 38/39
# MAGIC 
# MAGIC 8/30/2020
# MAGIC Solution Architect Coding Assignment Final - Databricks
# MAGIC when trying to achieve better accuracy without overfitting the model. For demonstration I chose to add grids for maxDepth and maxIter identical to the reference document to avoid long run times. Many different hyperparamters should be tested in order to determine the optimal set of values.
# MAGIC When using CrossValidator MLlib will automatically track trials in MLflow. I installed the PyPI package mlflow so that this behavior is enabled automatically and the results can be visualized in the MLflow user interface. The RMSE for the tuned model is lower as expected.

# COMMAND ----------

