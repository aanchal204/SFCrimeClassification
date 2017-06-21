import java.io._
import java.time.format.DateTimeFormatter

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.sql.{SQLContext}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier, OneVsRest}
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.{GBTClassifier}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier

object SfCrimeClassification {
    
    val enrichTime = (df: DataFrame) => {
    def dateUDF = udf { (timestamp: String) =>
      val timestampFormatter = DateTimeFormatter.ofPattern("YYYY-MM-dd HH:mm:ss")
      val dateFormat = DateTimeFormatter.ofPattern("YYYY-MM-dd")
      val time = timestampFormatter.parse(timestamp)
      dateFormat.format(time)
    }

    df
      .withColumn("HourOfDay", hour(col("Dates")))
      .withColumn("Month", month(col("Dates")))
      .withColumn("Year", year(col("Dates")))
  }

  def main(args: Array[String]) {
      //val t1 = System.nanoTime
    if (args.length < 1) {
      println("File path must be passed. " + args.length)
      System.exit(-1)
    }
    val trainFilePath = args(0)
    val conf = new SparkConf().setAppName("SfCrimeClassification")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

      //stop the info messages
      sc.setLogLevel("OFF")
      
      val trainData = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true").option("inferSchema", "true").load(trainFilePath)
      print("Done")
      /**
       * Training Phase
       */
      
    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = trainData.randomSplit(Array(0.7, 0.3))
      
    val categoryIndexer = new StringIndexer().setInputCol("Category")
      .setOutputCol("label")
      
      // feature engineering
    val enrichFunctions = List(enrichTime)
    val Array(enrichedTrainDF, enrichedTestDF) =
      Array(trainingData, testData) map (enrichFunctions reduce (_ andThen _))
      
    val dayOfWeekIndexer = new StringIndexer().setInputCol("DayOfWeek")
      .setOutputCol("DayOfWeekIndex")
    val pdDistrictIndexer = new StringIndexer().setInputCol("PdDistrict")
      .setOutputCol("PdDistrictIndex")
      
    val dayOfWeekEncoder = new OneHotEncoder().setInputCol("DayOfWeekIndex")
      .setOutputCol("DayOfWeekBinaryVector")
      
    val pdDistrictEncoder = new OneHotEncoder().setInputCol("PdDistrictIndex")
      .setOutputCol("PdDistrictBinaryVector")
      
    val vectorAssembler = new VectorAssembler().setInputCols(Array("DayOfWeekIndex",
      "PdDistrictBinaryVector", "X", "Y","HourOfDay","Month","Year")).setOutputCol("rowFeatures")
      
    val featureScaler = new StandardScaler().setInputCol("rowFeatures")
      .setOutputCol("features")
      
    //decision tree classifier
      val decisionTreeClassifier = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setImpurity("entropy")
      
      //logistic regression
    val logisticRegressionClassifier = new LogisticRegression().setMaxIter(200).setRegParam(0.001).setLabelCol("label").setFeaturesCol("features")
      
    //random forest classifier  
    val randomForestClassifier = new RandomForestClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setNumTrees(50)
      
      //Multi layer Perceptron classifier
      
      // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](4, 5, 4, 39)

    // create the trainer and set its parameters
    val multiLayerPerceptronClassifier = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
      
      /*
      Currently GBT only supports binary labels
    //gradient boosted tree classifier
      val gradientBoostedTreeClassifier = new GBTClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setMaxIter(10)
      */
      
    val trainPipeline = new Pipeline().setStages(Array(categoryIndexer, dayOfWeekIndexer,
      pdDistrictIndexer, dayOfWeekEncoder, pdDistrictEncoder, vectorAssembler, featureScaler, logisticRegressionClassifier))
      
    val model = trainPipeline.fit(enrichedTrainDF) 
      
      // Make predictions.
    val predictions = model.transform(enrichedTestDF)

    // Select example rows to display.
    predictions.select("prediction","label", "features").show(15)

      // Select (prediction, true label) and compute test error. - accuracy
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
      
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))
      
      //val duration = (System.nanoTime - t1) / 1e9d
      //println(duration)
     /* 
      // Select (prediction, true label) and compute test error. - weighted precision
    val WPevaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")
      
    val weightedPrecision = evaluator.evaluate(predictions)
    println("Weighted Precision = " + weightedPrecision)
      
      // Select (prediction, true label) and compute test error. - weighted recall
    val WRevaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("weightedRecall")
      
    val weightedRecall = evaluator.evaluate(predictions)
    println("Weighted Recall = " + weightedRecall) */

  }
}