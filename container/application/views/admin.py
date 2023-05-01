import io
from flask import Blueprint, render_template, request, redirect, flash,  url_for
from werkzeug.utils import secure_filename
import traceback
import csv
import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as f
from pyspark.sql import SQLContext
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import f1_score
admin = Blueprint('admin', __name__, url_prefix='/')

# sr2484 | Apr 7
@admin.route("/", methods=["GET","POST"])
def importCSV():
    val = 0
    testvalue = ""
    predict = ""
    args = {**request.args}
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file', "warning")
            return redirect(request.url)
        # TODO importcsv-1 check that it's a .csv file, return a proper flash message if it's not
        filename = str(file.filename)
        if filename[-4:] != ".csv":
            flash(f'File {filename} provided is not a csv file', 'danger')
            return redirect(request.url)
        file.save(filename)
        # with open(filename, 'wb') as f:
        #   f.write(file)
        print("File Saved")
        print("Create Model")

        print("Starting")
        APP_NAME = "Wine_Quality_Prediction"
        conf = SparkConf().setAppName(APP_NAME)
        sc = SparkContext(conf=conf)
        sqlContext = SQLContext(sc)
        spark = SparkSession.builder.appName(APP_NAME).getOrCreate()

        columns = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']
        print("Reading file")
        df = spark.read.option("delimiter",";").option("header","true").csv(filename)
        df = df.toDF('fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality')
        print("Testing Data")
        df.show()
        df.describe()

        df = df.toPandas()
        for item in columns:
            df[item] = pd.to_numeric(df[item])
        df.info()
        df = pd.DataFrame(df, columns=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality'])
        
        df = sqlContext.createDataFrame(df)
        columns = columns[:-1]
        assemblers = VectorAssembler(inputCols=columns, outputCol="features")
        test_data = assemblers.transform(df)

        path = './views/model/'

        rf_model = RandomForestClassificationModel.load(path)
        print("Loaded model")

        predictions = rf_model.transform(test_data)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="quality", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("Accuracy of the results = %g " % (accuracy))
        predictpandas = predictions.select('fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality','prediction').toPandas()
        predict ="""<h3> Table View </h3>"""
        predict += (predictpandas.to_html())
        print("f1 Score")
        val = f1_score(predictpandas["quality"], predictpandas["prediction"], average='micro')
        
        spark.stop()
        sc.stop()

    return render_template("upload.html", val=val, predict=predict)
