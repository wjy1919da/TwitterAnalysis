package scala.twitter_sentiment

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, LabeledPoint, StringIndexer, VectorAssembler, Word2Vec}
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import scala.twitter_sentiment.naiveBayes.SparkNaiveBayesModelCreator
import scala.twitter_sentiment.preprocess.PreProcess
import scala.twitter_sentiment.utils.{PropertiesLoader, StopwordsLoader}
import scala.twitter_sentiment.word2vec.SparkWord2VecNaiveBayesModelCreator

object TwitterSentimentTrainApplication {

  def main(args: Array[String]): Unit = {
    //initialized spark environment
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val sparkConf = new SparkConf()
      .setAppName("TwitterSentiment")
      .setMaster("local[*]")
      .set("spark.executor.processTreeMetrics.enabled", "false")
    val sc = new SparkContext(sparkConf)

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._
    //stop words load
    val stopWordsList = sc.broadcast(StopwordsLoader.loadStopWords(PropertiesLoader.nltkStopWords))
    val twittersDF = PreProcess.loadSentiment140File(spark, PropertiesLoader.sentiment140TrainingFilePath)
    twittersDF.show(10)
    println("dataset size : "+twittersDF.count())
    println("SparkNaiveBayesModelCreator : ")
   SparkNaiveBayesModelCreator.createAndSaveModel(sc,twittersDF,stopWordsList)
    println("validateAccuracyOfModel : ")
    SparkNaiveBayesModelCreator.validateAccuracyOfModel(sc,twittersDF,stopWordsList)
    spark.stop
  }


}
