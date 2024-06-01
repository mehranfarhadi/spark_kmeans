from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler

# Initialize Spark session
spark = SparkSession.builder.appName("KMeansClustering").getOrCreate()

# Load data
data = spark.read.csv("data.csv", header=True, inferSchema=True)

# Assemble features into a vector
assembler = VectorAssembler(inputCols=data.columns, outputCol="features")
feature_data = assembler.transform(data)

# Train a KMeans model
kmeans = KMeans().setK(3).setSeed(1)
model = kmeans.fit(feature_data)

# Make predictions
predictions = model.transform(feature_data)

# Evaluate clustering
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette with squared euclidean distance = {silhouette}")

# Show the result
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# Stop the Spark session
spark.stop()
