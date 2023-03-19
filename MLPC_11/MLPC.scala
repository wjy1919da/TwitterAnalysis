package MLPC_11
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession

object UtilityForSparkSession {
  def mySession: SparkSession = {
    val spark = SparkSession.builder.appName("MultilayerPerceptronClassifier").master("local[*]").config("spark.sql.warehouse.dir", "E:/Exp/").getOrCreate
    spark
  }
}

object MLPC {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val spark = UtilityForSparkSession.mySession
    val path = "E:/hw/input/Dataset.data"
    val dataFrame = spark.read.format("libsvm").load(path)
    val splits = dataFrame.randomSplit(Array[Double](0.8, 0.2), 12345L)
    val train = splits(0)
    val test = splits(1)
  }
}
