package scala.twitter_sentiment.word2vec

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, HashingTF, IDF, Word2Vec}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.twitter_sentiment.naiveBayes.SparkNaiveBayesModelCreator.{transformFeatures, validateAccuracyOfModel}
import scala.twitter_sentiment.preprocess.PreProcess

object SparkWord2VecNaiveBayesModelCreator {

  def createAndSaveWord2VecModel(spark: SparkSession,tweetsDF:DataFrame, stopWordsList: Broadcast[List[String]]): Unit = {
    import spark.implicits._
    val standerWordsDf = tweetsDF.mapPartitions { iter =>
      iter.map { case Row(polarity:Int, status:String) =>
        {
          val rrr =  PreProcess.getBarebonesTweetText(status, stopWordsList.value)
          (polarity ,rrr)
        }
      }
    }

    val twitterRawDf = standerWordsDf.withColumnRenamed("_1", "polarity")
      .withColumnRenamed("_2","status")
    twitterRawDf.show(10)

    //twitterRawDf.filter("status.length = 0").show(10)
    val Array(trainData,testData) = twitterRawDf.randomSplit(Array(0.7, 0.3))

    var hashingTF = new HashingTF().setNumFeatures(1048576).setInputCol("status").setOutputCol("rawFeatures")
    var featurizedData = hashingTF.transform(trainData)
    println("featurizedData")
    featurizedData.show(10)
    featurizedData.select($"polarity", $"status", $"rawFeatures").show(10)
    var idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    var idfModel = idf.fit(featurizedData)
    var rescaledData = idfModel.transform(featurizedData)
    println("IDF rescaledData：")
    rescaledData.show(10)

    val LabeledPointData = rescaledData.select($"polarity",$"features").rdd.map{
       case Row(polarity:Int, features:SparseVector) =>
        LabeledPoint(polarity, Vectors.dense(features.toArray))
    }
    println("LabeledPointData: ")
    LabeledPointData.take(5).foreach(println)

    val naiveBayesModel= NaiveBayes.train(LabeledPointData, lambda = 1.0, modelType = "multinomial")

    var testfeaturizedData = hashingTF.transform(testData)
    var testrescaledData = idfModel.transform(testfeaturizedData)
    var testDataRdd = testrescaledData.select($"polarity",$"features").map {
      case Row(polarity: Int, features: SparseVector) =>
        LabeledPoint(polarity, Vectors.dense(features.toArray))
    }

    //对测试数据集使用训练模型进行分类预测
//    val testpredictionAndLabel = testDataRdd.map(p => (naiveBayesModel.predict(p.features), p.label))
//
//    //统计分类准确率
//    var testaccuracy = 1.0 * testpredictionAndLabel.filter(x => x._1 == x._2).count() / testDataRdd.count()
//    println("output5：")
//    println(testaccuracy)
//    word2VecResult.select("polarity","feature").rdd.map {
//      case Row(polarity:Int,feature:Vector)=>
//        LabeledPoint(polarity, Vectors.dense(feature.toArray)
//    }
//    val naiveBayesModel: NaiveBayesModel = NaiveBayes.train(labeledRDD, lambda = 1.0, modelType = "multinomial")
//    //naiveBayesModel.save(sc, PropertiesLoader.naiveBayesModelPath)
//    validateAccuracyOfModel(spark, stopWordsList,naiveBayesModel)
  }
}
