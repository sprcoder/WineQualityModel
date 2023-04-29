from flask import Blueprint, render_template, request, redirect, flash,  url_for
import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as f
from pyspark.sql import SQLContext
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os
from sklearn.metrics import f1_score
model = Blueprint('model', __name__, url_prefix='/')
        
@model.route("/train", methods=["GET"])
def Accuracy_getter():
    print("starting")
    conf = (SparkConf().setAppName("SparkByExamples.com"))
    print("Starting")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    sqlContext = SQLContext(sc)
    client = boto3.client('s3')
    filename = request.args.get("filename","")
    columns = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']
    print("Reading file")
    df = spark.read.option("delimiter",";").option("header","true").csv(filename)
    df = df.toDF('fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality')
    print("Testing Data")
    df.show()

    df = df.toPandas()
    for item in columns:
        df[item] = pd.to_numeric(df[item])
    df = pd.DataFrame(df, columns=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality'])
    print(df.describe())
    df = self.sqlContext.createDataFrame(df)
    columns = columns[:-1]
    assemblers = VectorAssembler(inputCols=columns, outputCol="features")
    transOutput = assemblers.transform(df)

    testData = transOutput.select("features","quality")
    testData.show()

    path = '/rfmodel'
    print(path)
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
    val = f1_score(predictpandas["quality"], predictpandas["prediction"], average='micro')
    return render_template("upload.html", val=val)

