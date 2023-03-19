package scala.twitter_sentiment.utils

import com.typesafe.config.{Config, ConfigFactory}

/**
  * Exposes all the key-value pairs as properties object using Config object of Typesafe Config project.
  */
object PropertiesLoader {
  private val conf: Config = ConfigFactory.load("application.conf")
  val sentiment140TrainingFilePath = conf.getString("SENTIMENT140_TRAIN_DATA_PATH")
  val sentiment140TestingFilePath = conf.getString("SENTIMENT140_TEST_DATA_PATH")
  val nltkStopWords = conf.getString("STOPWORDS_FILE_NAME ")
  val naiveBayesModelPath = conf.getString("NAIVEBAYES_MODEL_PATH")
  val modelAccuracyPath = conf.getString("NAIVEBAYES_MODEL_ACCURACY_PATH ")
  val sentiment140TrainingSaveFilePath = conf.getString("SENTIMENT140_TESTSAVE_DATA_PATH ")
}