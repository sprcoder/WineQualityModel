import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.sql import functions as f
from pyspark.sql import SQLContext
from pyspark.ml.classification import RandomForestClassifier
import boto3
import io
import os
import tarfile

#Spark session creation
conf = (SparkConf().setAppName("SparkByExamples.com"))
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
spark = SparkSession(sc)
sqlContext = SQLContext(sc)

#Accessing S3 and getting training data
client = boto3.client('s3')
response = client.get_object(Bucket='wine-model', Key='TrainingDataset.csv')
file = response["Body"].read().decode('utf-8')
columns = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']

# Converting string data to dataframe
dfpd = pd.read_csv(io.StringIO(file), sep=";", names=columns)
dfpd = dfpd.drop(labels = 0, axis = 0)

# Sampling dataset to increase data for training
sample = dfpd.sample(frac=0.40)
dfpd = dfpd.append(sample)

# Converting pandas to dataframe
df = spark.createDataFrame(dfpd)
df = df.toDF('fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality')
print("Training Dataset")
df.show()
df = df.toPandas()
for item in columns:
    df[item] = pd.to_numeric(df[item])
df = pd.DataFrame(df, columns=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol', 'quality'])
print(df.describe())
df = sqlContext.createDataFrame(df)

#Creating a assembler to convert it to vectory type
columns = columns[:-1]
assemblers = VectorAssembler(inputCols=columns, outputCol="features")
transOutput = assemblers.transform(df)

trainingData = transOutput.select("features","quality")


"""Random Forest Classifier Implementation"""
rf_classifier = RandomForestClassifier(labelCol="quality", featuresCol="features", maxDepth=6, numTrees=30, impurity="gini").fit(trainingData)

# Save model to hdfs
print(rf_classifier)
path = 'hdfs:///model/'
os.system("hadoop fs -rm -r /model/")
os.system("hadoop fs -mkdir /model")
rf_classifier.write().overwrite().save(path)

# Move model to local file system
os.system("rm -r ./modell")
os.system("mkdir ./modell")
os.system("hadoop fs -get /model/ /home/ec2-user/modell/")

# Zip the model to a single file
with tarfile.open("MLModel.tar.gz", "w:gz") as tarhandle:
    for root, dirs, files in os.walk('./modell/'):
        for f in files:
            tarhandle.add(os.path.join(root, f))

# Upload the model to S3 bucket
client.upload_file(
    Filename="MLModel.tar.gz",
    Bucket='wine-model',
    Key='MLModel.tar.gz'
)

print("Successfully uploaded the trained model to S3 Bucket")