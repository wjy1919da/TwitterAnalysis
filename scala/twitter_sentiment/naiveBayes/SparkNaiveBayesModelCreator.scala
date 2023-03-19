package scala.twitter_sentiment.naiveBayes

import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

import scala.twitter_sentiment.preprocess.PreProcess
import scala.twitter_sentiment.preprocess.PreProcess.replaceNewLines
import scala.twitter_sentiment.utils.{PropertiesLoader, SQLContextSingleton}

object SparkNaiveBayesModelCreator {

  def createAndSaveModel(sc: SparkContext,tweetsDF:DataFrame, stopWordsList: Broadcast[List[String]]): Unit = {

    val labeledRDD = tweetsDF.select("polarity", "status").rdd.map {
      case Row(polarity: Int, tweet: String) =>
        val tweetInWords: Seq[String] = PreProcess.getBarebonesTweetText(tweet, stopWordsList.value)
          LabeledPoint(polarity, transformFeatures(tweetInWords))//词转特征向量
    }
    labeledRDD.cache()
    labeledRDD.take(10).foreach(println)
    val naiveBayesModel = NaiveBayes.train(labeledRDD, lambda = 1.0, modelType = "multinomial")//拉普拉斯平滑系数
    naiveBayesModel.save(sc, PropertiesLoader.naiveBayesModelPath)
    //validateAccuracyOfModel(spark, stopWordsList,naiveBayesModel)
  }

  def validateAccuracyOfModel(sc: SparkContext, tweetsDF:DataFrame,stopWordsList: Broadcast[List[String]]): Unit = {
    //
    val naiveBayesModel = NaiveBayesModel.load(sc, PropertiesLoader.naiveBayesModelPath)//加载模型

    val actualVsPredictionRDD = tweetsDF.select("polarity", "status").rdd.map {
      case Row(polarity: Int, tweet: String) =>
        val tweetText = replaceNewLines(tweet)
        val tweetInWords: Seq[String] = PreProcess.getBarebonesTweetText(tweetText, stopWordsList.value)
        (polarity.toDouble,
          naiveBayesModel.predict(transformFeatures(tweetInWords)),
          tweetText)
    }//????
    val accuracy = 100.0 * actualVsPredictionRDD.filter(x => x._1 == x._2).count() / tweetsDF.count()
    println(f"""\n\t<==******** Prediction accuracy compared to actual: $accuracy%.2f%% ********==>\n""")
    saveAccuracy(sc, actualVsPredictionRDD)
  }//交叉验证？？？训练集和测试集的划分

  def saveAccuracy(sc: SparkContext, actualVsPredictionRDD: RDD[(Double, Double, String)]): Unit = {
    val sqlContext = SQLContextSingleton.getInstance(sc)
    import sqlContext.implicits._
    val actualVsPredictionDF = actualVsPredictionRDD.toDF("Actual", "Predicted", "Text")
    actualVsPredictionDF.coalesce(1).write
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("delimiter", "\t")
      // Compression codec to compress while saving to file.
      .option("codec", classOf[GzipCodec].getCanonicalName)
      .mode(SaveMode.Append)
      .save(PropertiesLoader.modelAccuracyPath)
  }

  /**
   * Transforms features to Vectors.
   *
   * @param tweetText -- Complete text of a tweet.
   * @return Vector
   */
  def transformFeatures(tweetText: Seq[String]): linalg.Vector = {
    val hashingTF = new HashingTF()
    hashingTF.transform(tweetText)
  }//把字符转化为词频向量
}

