package scala.twitter_sentiment

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, Word2Vec}
import org.apache.spark.sql.SparkSession

object test {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val spark = SparkSession.builder().appName(this.getClass.getSimpleName).master("local").getOrCreate()
    val parsedRDD = spark.sparkContext.textFile("twitter_sentiment_data/training/test.csv").map(line => {
      val arr = line.split(" ")
      if (arr.length == 4) {
        (arr(3), arr(2).split(","))
      } else {
        ("", "".split(","))
      }
    })
    val msgDF = spark.createDataFrame(parsedRDD).toDF("label", "message")
    msgDF.printSchema()
    msgDF.show()
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(msgDF)

    val word2Vec = new Word2Vec().setInputCol("message").setOutputCol("features").setVectorSize(2).setMinCount(1)

    //    val layers = Array[Int](2, 250, 500, 200)
//    val mlpc = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(512).setSeed(1234L)
//      .setMaxIter(128)
//      .setFeaturesCol("features")
//      .setLabelCol("indexedLabel")
//      .setPredictionCol("prediction")
//
//    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
//
//    val Array(trainingData, testData) = msgDF.randomSplit(Array(0.8, 0.2))
//    val pipeline = new Pipeline().setStages(Array(labelIndexer, word2Vec, mlpc, labelConverter))
//    val model = pipeline.fit(trainingData)
//    val predictionResultDF = model.transform(testData)
//    //below 2 lines are for debug use
//    predictionResultDF.printSchema
//    predictionResultDF.select("message", "label", "predictedLabel").show(30)
//    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("precision")
//    val predictionAccuracy = evaluator.evaluate(predictionResultDF)
//    println("Testing Accuracy is %2.4f".format(predictionAccuracy * 100) + "%")
//    spark.stop
//  }
//
//  val labelAndPreds = test.map { point =>
//    val prediction = model.predict(point.features)
//    (prediction, point.label)
//  }
//
//  // Get evaluation metrics.
//  val metrics = new BinaryClassificationMetrics(labelAndPreds)
//  val auROC = metrics.areaUnderROC()
}}
