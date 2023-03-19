package scala.twitter_sentiment.preprocess

import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

object PreProcess {

  def loadSentiment140File(spark: SparkSession, sentiment140FilePath: String): DataFrame = {
    val tweetsDF = spark.read.option("header", "true")
      .format("com.databricks.spark.csv")
      .option("inferSchema", "true")
      .load(sentiment140FilePath)
      .toDF("id", "polarity", "user", "status")

    tweetsDF.drop("id").drop("user")
  }

  def normalizeMLlibSentiment(sentiment: Double) = {
    sentiment match {
      case x if x == 0 => 1 // negative
      case x if x == 2 => 2 // neutral
      case x if x == 4 => 3 // positive
      case _ => 0 // if cant figure the sentiment, term it as neutral
    }
  }

  /**
   * Strips the extra characters in tweets. And also removes stop words from the tweet text.
   *
   * @param tweetText     -- Complete text of a tweet.
   * @param stopWordsList -- Broadcast variable for list of stop words to be removed from the tweets.
   * @return Seq[String] after removing additional characters and stop words from the tweet.
   */
  def getBarebonesTweetText(tweetText: String, stopWordsList: List[String]): Seq[String] = {
    //Remove URLs, RT, MT and other redundant chars / strings from the tweets.
    tweetText.toLowerCase()
      //.replaceAll("\n", "")
      //.replaceAll("rt\\s+", "")
      .replaceAll("\\s+@\\w+", "")
      .replaceAll("@\\w+", "")
      //.replaceAll("\\s+#\\w+", "")
      //.replaceAll("#\\w+", "")
      .replaceAll("(?:https?|http?)://[\\w/%.-]+", "")
      .replaceAll("(?:https?|http?)://[\\w/%.-]+\\s+", "")
      .replaceAll("(?:https?|http?)//[\\w/%.-]+\\s+", "")
      .replaceAll("(?:https?|http?)//[\\w/%.-]+", "")
      .split("\\W+")//以空格为间隔，把字符串分割成字符串数组
      //.filter(_.matches("^[a-zA-Z]+$"))//过滤掉所有数字，只留下字母
      //.filter(x=>x.length!=1)//过滤单个字母的词
      //.filter(!stopWordsList.contains(_))//过滤掉stop words
  }
  /**
   * Remove new line characters.
   *
   * @param tweetText -- Complete text of a tweet.
   * @return String with new lines removed.
   */
  def replaceNewLines(tweetText: String): String = {
    tweetText.replaceAll("\n", "")
  }

  val hashingTF = new HashingTF()

  /**
   * Transforms features to Vectors.
   *
   * @param tweetText -- Complete text of a tweet.
   * @return Vector
   */

}
