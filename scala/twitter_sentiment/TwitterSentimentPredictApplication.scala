package scala.twitter_sentiment

import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import scala.twitter_sentiment.naiveBayes.SparkNaiveBayesModelCreator
import scala.twitter_sentiment.preprocess.PreProcess
import scala.twitter_sentiment.preprocess.PreProcess.{getBarebonesTweetText, normalizeMLlibSentiment}
import scala.twitter_sentiment.utils.{PropertiesLoader, StopwordsLoader}

object TwitterSentimentPredictApplication {

  def TextToEmotion(polarity:Int):String={
    polarity match {
      case x if x == 1 => "(ô◠ô)" // negative
      case x if x == 2 => "(ô-ô)" // neutral
      case x if x == 3 => "(ô‿ô)" // positive
      case _ => "0" // if cant figure the sentiment, term it as neutral
    }
  }
  def computeSentiment(text: String, stopWordsList: Broadcast[List[String]], model: NaiveBayesModel): Int = {
    val tweetInWords: Seq[String] = getBarebonesTweetText(text, stopWordsList.value)
    val polarity = model.predict(SparkNaiveBayesModelCreator.transformFeatures(tweetInWords))
    normalizeMLlibSentiment(polarity)
  }
//  def predictSentiment(tweets: String,stopWordsList: Broadcast[List[String]],naiveBayesModel): String = {
//    val tweetText = PreProcess.replaceNewLines(tweets)
//    val (corenlpSentiment, mllibSentiment) =
//              (CoreNLPSentimentAnalyzer.computeWeightedSentiment(tweetText),
//          computeSentiment(tweetText, stopWordsList, naiveBayesModel))
//  }
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
    //stop words load
    val stopWordsList = sc.broadcast(StopwordsLoader.loadStopWords(PropertiesLoader.nltkStopWords))

    val naiveBayesModel = NaiveBayesModel.load(sc, PropertiesLoader.naiveBayesModelPath)//加载模型
    val tweet = "happy" ;

    val mllibSentiment = computeSentiment(tweet,stopWordsList,naiveBayesModel)
    //val corenlpSentiment = CoreNLPSentimentAnalyzer.computeWeightedSentiment(tweet)
    println("mllibSentiment "+mllibSentiment)
    //println("corenlpSentiment "+corenlpSentiment)
    println(TextToEmotion(mllibSentiment))
    spark.stop
  }


}
