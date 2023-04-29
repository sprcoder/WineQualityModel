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
import sys
from sklearn.metrics import f1_score

conf = (SparkConf().setAppName("SparkByExamples.com"))
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
spark = SparkSession(sc)
sqlContext = SQLContext(sc)

path = ""
iss3 = False
if len(sys.argv) > 1 and sys.argv[1] != None:
    path = sys.argv[1]
    print(path)
if path == "s3":
    bucketname = input("Provide s3 bucket name: ")
    objectkey = input("Provide s3 object key: ")
    if objectkey[-4:] != ".csv":
        print("provided object should be a csv file")
        print("Exiting the program")
        exit()
    else:
        iss3 = True
elif not os.path.isfile(path):
    print("Provided path is not pointing to a file. \nExiting the program")
    exit()

columns = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']
client = boto3.client('s3')
if iss3:
    response = client.get_object(Bucket='wine-model', Key='ValidationDataset.csv')
    file = response["Body"].read().decode('utf-8')
    dfpd = pd.read_csv(io.StringIO(file), sep=";", names=columns)
    dfpd = dfpd.drop(labels = 0, axis = 0)
    df = spark.createDataFrame(dfpd)
else:
    df = spark.read.option("delimiter",";").option("header","true").csv(path)

df = df.toDF('fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality')
print("Testing Data")
df.show()

df = df.toPandas()
for item in columns:
    df[item] = pd.to_numeric(df[item])
df = pd.DataFrame(df, columns=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality'])
print(df.describe())
df = sqlContext.createDataFrame(df)
columns = columns[:-1]
assemblers = VectorAssembler(inputCols=columns, outputCol="features")
transOutput = assemblers.transform(df)

testData = transOutput.select("features","quality")
testData.show()

# Download the model
os.system("mkdir data")
client.download_file('wine-model', 'MLModel.tar.gz', 'data/MLModel.tar.gz')
my_tar = tarfile.open('data/MLModel.tar.gz')
os.system("rm -r ./extractedl")
os.system("mkdir extractedl")

# Extract the compressed model
my_tar.extractall('./extractedl/')

# Move the model to hdfs file system
path = './extractedl/modell/model/'

rf_model = RandomForestClassificationModel.load(path)
print("Loaded model")
print(rf_model)

predictions = rf_model.transform(testData)
predictions.show()
evaluator = MulticlassClassificationEvaluator(
    labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy of the results = %g " % (accuracy))
predictpandas = predictions.toPandas()
print("f1 Score")
print(f1_score(predictpandas["quality"], predictpandas["prediction"], average='micro'))

