from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Create a Spark session
spark = SparkSession.builder.appName("LinearRegressionExample").getOrCreate()

# Load your data
data = spark.read.format("csv").option("header", "true").load("hdfs://localhost:54310/user/hduser/hdfs-directory/daily_file.csv")

# Example: Drop missing values
data = data.na.drop(subset=["Day", "Confirmed", "Recovered", "Deaths"])

# Selecting features and converting string columns to numeric types
selected_features = data.select(
    col("Day").cast("double"),
    col("Confirmed").cast("double"),
    col("Deaths").cast("double"),
    col("Recovered").cast("double"),
    col("Active").cast("double"),
    col("New Cases").cast("double"),
    col("New Deaths").cast("double"),
    col("New recovered").cast("double")
)

# Assemble features into a single vector column and overwrite the existing column
assembler = VectorAssembler(inputCols=["Day", "Confirmed", "Recovered", "Deaths"], outputCol="features")
data_assembled = assembler.transform(selected_features).select("Day", "Confirmed", "Recovered", "Deaths", "features")

# Split the data into training and test sets
(training_data, test_data) = data_assembled.randomSplit([0.8, 0.2], seed=123)

# Create a linear regression model
lr = LinearRegression(featuresCol="features", labelCol="Deaths")

# Create a pipeline
pipeline = Pipeline(stages=[lr])

# Train the model
model = pipeline.fit(training_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Display predictions vs. actual values
predictions.select("Deaths", "prediction").show()

