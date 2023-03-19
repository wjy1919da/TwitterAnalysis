package scala.twitter_sentiment

import com.google.gson.Gson
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{Row, SparkSession}

import scala.twitter_sentiment.preprocess.PreProcess
import scala.twitter_sentiment.preprocess.PreProcess.replaceNewLines
import scala.twitter_sentiment.utils.{PropertiesLoader, StopwordsLoader}

object SaveTest {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val sparkConf = new SparkConf().setAppName("TwitterModelSave").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    val spark: SparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._

    //stop words load
    val stopWordsList = sc.broadcast(StopwordsLoader.loadStopWords(PropertiesLoader.nltkStopWords))
    val twittersDF = PreProcess.loadSentiment140File(spark, PropertiesLoader.sentiment140TrainingSaveFilePath)
    twittersDF.show(10)

    val twitterDs = twittersDF.mapPartitions { iter =>

      iter.map { case Row(polarity:String, status:String) =>
        (polarity ,PreProcess.getBarebonesTweetText(status, stopWordsList.value).toList.mkString(" ") )
       }
    }
    twitterDs.cache()

    val twitterDf = twitterDs.withColumnRenamed("_1", "polarity").withColumnRenamed("_2","status")
    twitterDf.show(10)
    val text = twitterDf
    text.coalesce(1).write.mode("Append").format("csv").save("twitter")

  }
}
