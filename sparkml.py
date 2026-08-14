from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import happybase

# Step 1: Create a Spark session
spark = SparkSession.builder.appName("HospitalReadmissionML").enableHiveSupport().getOrCreate()

# Step 2: Load hospital data from Hive into Spark DataFrame
hospital_df = spark.sql("SELECT Age, Gender, Condition, Procedure, Cost, Length_of_Stay, Readmission, Satisfaction FROM hospital")

# Step 3: Handle null values by either dropping or filling them
hospital_df = hospital_df.na.drop()

# Step 3.5: Convert categorical columns into numerical ones and use one hot encoding
gender_indexer = StringIndexer(inputCol="Gender",outputCol="GenderIndex",handleInvalid="keep")

condition_indexer = StringIndexer(inputCol="Condition",outputCol="ConditionIndex",handleInvalid="keep")

procedure_indexer = StringIndexer(inputCol="Procedure",outputCol="ProcedureIndex",handleInvalid="keep")

readmission_indexer = StringIndexer(inputCol="Readmission",outputCol="label")

indexed_df = gender_indexer.fit(hospital_df).transform(hospital_df)
indexed_df = condition_indexer.fit(indexed_df).transform(indexed_df)
indexed_df = procedure_indexer.fit(indexed_df).transform(indexed_df)
indexed_df = readmission_indexer.fit(indexed_df).transform(indexed_df)

encoder = OneHotEncoder(
    inputCols=["GenderIndex", "ConditionIndex", "ProcedureIndex"],
    outputCols=["GenderVec", "ConditionVec", "ProcedureVec"]
)

encoded_df = encoder.fit(indexed_df).transform(indexed_df)

# Step 4: Prepare the data for MLlib by assembling features into a vector
assembler = VectorAssembler(
    inputCols=[
        "Age",
        "GenderVec",
        "ConditionVec",
        "ProcedureVec",
        "Cost",
        "Length_of_Stay",
        "Satisfaction"
    ],
    outputCol="features",
    handleInvalid="skip"
)

assembled_df = assembler.transform(encoded_df).select("features", "label")

# Step 5: Split the data into training and testing sets
train_data, test_data = assembled_df.randomSplit([0.7, 0.3])

# Step 6: Initialize and train a Logistic Regression model
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label"
)

lr_model = lr.fit(train_data)

# Step 7: Evaluate the model on the test data
predictions = lr_model.transform(test_data)

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)

# Step 8: Print the model performance metrics
#AUC
print("Hospital Readmission Prediction Results")
print("AUC:", auc)

#Accuracy
correct = predictions.filter(
    predictions.label == predictions.prediction
).count()

total = predictions.count()
accuracy = correct / total
print("Accuracy:", accuracy)

# Step 9: Write metrics to HBase with happybase

data = [
    ("hospital_metrics",
     "cf:auc",
     str(auc)),

    ("hospital_metrics",
     "cf:accuracy",
     str(accuracy))
]

# Function to write data to HBase inside each partition
def write_to_hbase_partition(partition):
    connection = happybase.Connection("master")
    connection.open()
    table = connection.table("my_table")

    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()

rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Step 10: Stop the Spark session
spark.stop()