import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.sql import functions as f
from pyspark.sql import SQLContext
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import boto3
import io
import os
import tarfile
from sklearn.metrics import f1_score

#Spark session creation
conf = (SparkConf().setAppName("SparkByExamples.com"))
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
spark = SparkSession(sc)
sqlContext = SQLContext(sc)

#Accessing S3 and getting validation data
client = boto3.client('s3')
response = client.get_object(Bucket='wine-model', Key='ValidationDataset.csv')
file = response["Body"].read().decode('utf-8')
columns = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']

# Converting string data to dataframe
dfpd = pd.read_csv(io.StringIO(file), sep=";", names=columns)
dfpd = dfpd.drop(labels = 0, axis = 0)

# Converting pandas to dataframe
df = spark.createDataFrame(dfpd)
df = df.toDF('fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality')
print("Validation Dataset")
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

validationData = transOutput.select("features","quality")

# Download the model
client.download_file('wine-model', 'MLModel.tar.gz', 'data/MLModel.tar.gz')
my_tar = tarfile.open('data/MLModel.tar.gz')
os.system("rm -r ./extractedl")
os.system("mkdir extractedl")

# Extract the compressed model
my_tar.extractall('./extractedl/')
os.system("hadoop fs -rm -r /extracted")
os.system("hadoop fs -mkdir /extracted")

# Move the model to hdfs file system
os.system("hadoop fs -put ./extractedl/ /extracted/")
path = 'hdfs:///extracted/extractedl/modell/model/'

rf_model = RandomForestClassificationModel.load(path)
print("Loaded model")
print(rf_model)

predictions = rf_model.transform(validationData)
predictions.show()
evaluator = MulticlassClassificationEvaluator(
    labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy of the model = %g " % (accuracy))
predictpandas = predictions.toPandas()
print("f1 Score")
print(f1_score(predictpandas["quality"], predictpandas["prediction"], average='micro'))