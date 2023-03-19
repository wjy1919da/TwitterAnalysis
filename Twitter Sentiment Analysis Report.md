### Twitter Sentiment Analysis NaiveBayes  Report

#### 1、全局设置

##### 1）、定义全局配置文件Application.conf

配置文件中定义twitter语料文件，模型文件、stopword文件的路径

```scala
SENTIMENT140_TRAIN_DATA_PATH = "twitter_sentiment_data/training/training.csv"
SENTIMENT140_TEST_DATA_PATH  = "twitter_sentiment_data/training/training.csv"
SENTIMENT140_TESTSAVE_DATA_PATH  = "twitter_sentiment_data/training/training.csv"
# The relative path to save the Naive Bayes Model of training data.
NAIVEBAYES_MODEL_PATH = "twitter_sentiment_data/Naivebayes/model"

# The relative path to save the accuracy of Naive Bayes Model after executing it with 
NAIVEBAYES_MODEL_ACCURACY_PATH = "twitter_sentiment_data/Naivebayes/accuracy/"

# Name of the file in the classpath [resources folder] which contains the stop words.
STOPWORDS_FILE_NAME=english

TWITTER_WORD_COUNT="twitter_sentiment_data/wordcount/switter_text.csv"
WORD2VEC_MODEL_PATH = "twitter_sentiment_data/word2vec/model"
LOGISTICREGRESSION_MODEL_PATH = "twitter_sentiment_data/logisticlegression/model"
```

##### 2）、定义全局工具类

定义 PropertiesLoader 类，加载application.conf中定义的文件路径

```scala
object PropertiesLoader {
  private val conf = ConfigFactory.load("application.conf")
  val sentiment140TrainingFilePath = conf.getString("SENTIMENT140_TRAIN_DATA_PATH")
  val sentiment140TestingFilePath = conf.getString("SENTIMENT140_TEST_DATA_PATH")
  val nltkStopWords = conf.getString("STOPWORDS_FILE_NAME")
  val naiveBayesModelPath = conf.getString("NAIVEBAYES_MODEL_PATH")
  val modelAccuracyPath = conf.getString("NAIVEBAYES_MODEL_ACCURACY_PATH")
  val sentiment140TrainingSaveFilePath = conf.getString("SENTIMENT140_TESTSAVE_DATA_PATH")
  val wordCountFilePath = conf.getString("TWITTER_WORD_COUNT")
  val word2vecModelFilePath = conf.getString("WORD2VEC_MODEL_PATH")
  val logisticRegressionModelPath = conf.getString("LOGISTICREGRESSION_MODEL_PATH")
}
```

##### 3）、定义stopwords加载类

读取资源文件夹下的stopword

```scala
object StopwordsLoader {
  def loadStopWords(stopWordsFileName: String): List[String] = {
    Source.fromInputStream(getClass.getResourceAsStream("/" + stopWordsFileName)).getLines().toList
  }
}
```

##### 4）、定义加载数据集全局类

加载twitter 语料文件，设置inferSchema为true，读取文件时自动转换数据类型，增加标题行

文件读取后只保留polarity与status列

```scala
def loadSentiment140File(spark: SparkSession, sentiment140FilePath: String): DataFrame = {
  val tweetsDF = spark.read.option("header", "true")
    .format("com.databricks.spark.csv")
    .option("inferSchema", "true")
    .load(sentiment140FilePath)
    .toDF("polarity", "id", "date", "query", "user", "status")

  tweetsDF.drop("id").drop("date").drop("query").drop("user")
}
```



##### 5）、以广播方式加载stopword

```scala
val stopWordsList = sc.broadcast(StopwordsLoader.loadStopWords(PropertiesLoader.nltkStopWords))
```

##### 6）、定义语料文件预处理方法

对语料文件按行将status列预处理，Remove URLs, RT, MT and other redundant chars，去掉stopword中包含的word

```scala
def getBarebonesTweetText(tweetText: String, stopWordsList: List[String]): Seq[String] = {
  //Remove URLs, RT, MT and other redundant chars / strings from the tweets.
  tweetText.toLowerCase()
    .replaceAll("\n", "")
    .replaceAll("rt\\s+", "")
    .replaceAll("\\s+@\\w+", "")
    .replaceAll("@\\w+", "")
    .replaceAll("\\s+#\\w+", "")
    .replaceAll("#\\w+", "")
    .replaceAll("(?:https?|http?)://[\\w/%.-]+", "")
    .replaceAll("(?:https?|http?)://[\\w/%.-]+\\s+", "")
    .replaceAll("(?:https?|http?)//[\\w/%.-]+\\s+", "")
    .replaceAll("(?:https?|http?)//[\\w/%.-]+", "")
    .split("\\W+")
    .filter(_.matches("^[a-zA-Z]+$"))
    //.filter(x=>x.length!=1)
    .filter(!stopWordsList.contains(_))
    //.map(x=>RemoveRepeatChar.removeRepeatChar(x))
  //val twweet = tweetList.map(x=>RemoveRepeatChar.removeRepeatChar(x))
  //twweet
}
```

#### 2、词频统计并绘制词云

对twitter语料文件预处理，并将得到的结果向量转换为字符串 PreProcess.getBarebonesTweetText(status, stopWordsList.value).toList.mkString(" ")

```scala
val twitterDs = twittersDF.mapPartitions { iter =>
  iter.map { case Row(polarity:Int, status:String) =>
    (polarity ,PreProcess.getBarebonesTweetText(status, stopWordsList.value).toList.mkString(" ") )
   }
}
```

获取status列并保存为文件

![1637981565288](twitter_image/1637981565288.png)

文件内容

![1637981734481](twitter_image/1637981734481.png)

wordclound程序读取switter_text.csv 并统计词频

![1637982368640](twitter_image/1637982368640.png)

使用python绘制词云图

![1637982620355](twitter_image/1637982620355.png)



#### 3、NaiveBayes  Analysis

##### 1）读取twitter语料文件

![1637983428056](twitter_image/1637983428056.png)

##### 2）保留polarity和status

![1637983547392](twitter_image/1637983547392.png)

##### 3）统计twitter语料文件条数

dataset size : 1600000

##### 3）定义transformFeatures方法

调用HashingTF 实现接收词条的集合然后把这些集合转化成固定长度的特征向量。这个算法在哈希的同时会统计各个词条的词频。采用默认的特征维度是1048576

```scala
val hashingTF = new HashingTF()
def transformFeatures(tweetText: Seq[String]): linalg.Vector = {

  hashingTF.transform(tweetText)
}
```

##### 4）语料文件转换为词向量

对twitter数据转换为rdd 并按行调用PreProcess.getBarebonesTweetText方法

对每一行调用transformFeatures转换为词向量

返回LabeledPoint格式返回

```scala
val labeledRDD = tweetsDF.select("polarity", "status").rdd.map {
  case Row(polarity: Int, tweet: String) =>
    val tweetInWords: Seq[String] = PreProcess.getBarebonesTweetText(tweet, stopWordsList.value)
      LabeledPoint(polarity, transformFeatures(tweetInWords))
}
```

###### ![1637979860306](twitter_image/1637979860306.png)

##### 5）生成 naiveBayesModel

```scala
val naiveBayesModel = NaiveBayes.train(labeledRDD, lambda = 1.0, modelType = "multinomial")
```

##### 6）模型文件保存

```
naiveBayesModel.save(sc, PropertiesLoader.naiveBayesModelPath)
```

![1637982831464](twitter_image/1637982831464.png)

##### 7）使用测试语料计算accuracy

加载保存的naiveBayesModel

```
val naiveBayesModel = NaiveBayesModel.load(sc, PropertiesLoader.naiveBayesModelPath)
```

测试文件预处理

```scala
val actualVsPredictionRDD = tweetsDF.select("polarity", "status").rdd.map {
  case Row(label: Int, tweet: String) =>
    val tweetText = replaceNewLines(tweet)
    val tweetInWords: Seq[String] = PreProcess.getBarebonesTweetText(tweetText, stopWordsList.value)
    (label.toDouble,
      naiveBayesModel.predict(transformFeatures(tweetInWords)),
      tweetText)
}
```

accuracy计算

```
val accuracy = 100.0 * actualVsPredictionRDD.filter(x => x._1 == x._2).count() / tweetsDF.count()
```

accuracy计算结果

<==******** Prediction accuracy compared to actual: 79.08% ********==>

#### 4、生成Word2VecMode以获取Synonyms

Word2VecMode 文件

![1637986590741](twitter_image/1637986590741.png)

##### 1）twitter数据预处理

```scala
val standerWordsDf = tweetsDF.mapPartitions { iter =>
  iter.map { case Row(polarity:Int, status:String) =>
    {
      val rrr =  PreProcess.getBarebonesTweetText(status, stopWordsList.value)
      (polarity ,rrr)
    }
  }
}
```

![1637985397361](twitter_image/1637985397361.png)

##### 2）定义word2vec model

```scala
val word2Vec = new Word2Vec()
  .setInputCol("status")
  .setOutputCol("features")
  .setVectorSize(300)
  .setMinCount(10);
```

##### 3）word2vec model生成

```scala
val model = word2Vec.fit(twitterRawDf)
```

##### 4）word2vec model 保存

```scala
model.write.overwrite().save(PropertiesLoader.word2vecModelFilePath)
```

![1637985195421](twitter_image/1637985195421.png)

```scala
val df = model.getVectors
df.show(10)
```

![1637986469721](twitter_image/1637986469721.png)

#### 5、naiveBayesModel predictor

##### 1）加载 naiveBayesModel文件

```scala
val naiveBayesModel = NaiveBayesModel.load(sc, PropertiesLoader.naiveBayesModelPath)
```

##### 2）加载val word2VecModel 

```scala
val word2VecModel = Word2VecMode.load(PropertiesLoader.word2vecModelFilePath)
```

##### 3）获取输入语句的Synonyms

```
val synonyms = word2VecModel.findSynonyms(tweet,10)
synonyms.show()
```

![1637984413360](twitter_image/1637984413360.png)

##### 4）将获得的Synonyms转换为词向量并预测此类的预测结果

```scala
val mllibSentiment = computeSentiment(tweet,stopWordsList,naiveBayesModel)
```

```scala
def computeSentiment(text: String, stopWordsList: Broadcast[List[String]], model: NaiveBayesModel): Int = {
  val tweetInWords: Seq[String] = getBarebonesTweetText(text, stopWordsList.value)
  val vector = SparkNaiveBayesModelCreator.transformFeatures(tweetInWords)
  val polarity = model.predict(vector)
  //model.predictProbabilities()
  val probabilities = model.predictProbabilities(vector)
  println(polarity+" , "+probabilities)
  normalizeMLlibSentiment(polarity)
}
```

预测结果

4.0 , [0.0022372966968223244,0.9977627033031775]

经预测结果转化为字符表情

```scala
println(TextToEmotion(mllibSentiment))
```

结果

mllibSentiment 3
(ô‿ô)

#### 6、logisticRegression 分析

![1637986849322](twitter_image/1637986849322.png)

##### 1）文件预处理

```scala
val standerWordsDf = tweetsDF.mapPartitions { iter =>
  iter.map { case Row(polarity:Int, status:String) =>
  {
    val rrr =  PreProcess.getBarebonesTweetText(status, stopWordsList.value)
    (polarity ,rrr)
  }
  }
}
```

##### 2）词频向量

```scala
 val hashingTF = new HashingTF().setNumFeatures(50000).setInputCol("status").setOutputCol("features")
    val has = hashingTF.transform(twitterRawDf)
```

##### 3）logisticRegression model 生成

```
val logisticRegression = new LogisticRegression().setMaxIter(20).setRegParam(0.1)
//
val model = logisticRegression.fit(has)
```

##### 4）模型保存

```scala
model.write.overwrite().save(PropertiesLoader.logisticRegressionModelPath)
```

##### 5）evaluator

```
val predictions = model.transform(has)

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
```

结果

Test accuracy = 0.7851375

#### 7 SVMWithSGD 分析

##### 1）预处理

```scala
val labeledRDD = twittersDF.select("polarity", "status").rdd.map {
  case Row(label: Int, tweet: String) =>
    val tweetInWords: Seq[String] = PreProcess.getBarebonesTweetText(tweet, stopWordsList.value)
    LabeledPoint(PreProcess.normalizeMLlibSVM(label), transformFeatures(tweetInWords))
}
```

##### 2）模型定义与生成

```scala
val model = SVMWithSGD.train(training, numIterations)
```

##### 3）评估

```scala
val scoreAndLabels = testTrain.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}
val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val auROC = metrics.areaUnderROC()
```

结果

Area under ROC = 0.749648701594588

#### 8、 python 神经网络 rnn

##### 1）文件读取

```python
dataset_filename = os.listdir("../input")[0]
dataset_path = os.path.join("..","input",dataset_filename)
print("Open file:", dataset_path)
df = pd.read_csv(dataset_path, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
```

![1637995228237](twitter_image/1637995228237.png)

##### 2）word2vec 

```python
w2v_model = gensim.models.word2vec.Word2Vec(vector_size=W2V_SIZE, 
                                            window=W2V_WINDOW, 
                                            min_count=W2V_MIN_COUNT, 
                                            workers=8)

w2v_model.build_vocab(documents)
```

```python
w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
```

##### 3） build model 

```python
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()
```

结果：

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 300, 300)          87125700  
_________________________________________________________________
dropout_1 (Dropout)          (None, 300, 300)          0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 100)               160400    
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 101       
=================================================================
Total params: 87,286,201
Trainable params: 160,501
Non-trainable params: 87,125,700
_________________________________________________________________
```

##### 4）compile model

```python
model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
```

##### 5）train

```python
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)
```

结果 耗时78多个小时

```
Train on 1152000 samples, validate on 128000 samples
Epoch 1/8
1152000/1152000 [==============================] - 1044s 906us/step - loss: 0.5077 - acc: 0.7477 - val_loss: 0.4647 - val_acc: 0.7793
Epoch 2/8
1152000/1152000 [==============================] - 1042s 904us/step - loss: 0.4814 - acc: 0.7659 - val_loss: 0.4591 - val_acc: 0.7830
Epoch 3/8
1152000/1152000 [==============================] - 1041s 904us/step - loss: 0.4743 - acc: 0.7703 - val_loss: 0.4559 - val_acc: 0.7858
```

##### 6） Evaluate

```
score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])
```

结果

```
ACCURACY: 0.791134375
LOSS: 0.4442952796936035
```



![1637995717978](twitter_image/1637995717978.png)

![1637995760354](twitter_image/1637995760354.png)

![1637995785059](twitter_image/1637995785059.png)

#### 9、 比较

| classifier | ACCURACY    |
| ---------- | ----------- |
| rnn        | 0.791134375 |
| svm        | 0.749648702 |
| lgrs       | 0.7851375   |
| NaiveBayes | 0.7908      |



![1637996194171](twitter_image/1637996194171.png)



lambda  accuracy
0.5  0.7647339350728902
1.0  0.7663687804405269
1.5  0.7669816143286973
2.0  0.767224139237844
2.5  0.7673627453341195
3.0  0.767389068607763
3.5  0.7673863824455252
4.0  0.7672754844087919
4.5  0.7672875317861075
5.0  0.7672204736199546
5.5  0.7672493011911011
6.0  0.7671865018005979



![1638091044747](twitter_image/1638091044747.png)

numIter = 20

regparam   accuracy 

0.1  0.7623154810705531
0.2  0.7622093579760869
0.3  0.7619929500971755
0.4  0.7616454489839234
0.5  0.7615497301144049
0.6  0.7613021095606504
0.7  0.7611106718216133
0.8  0.760875536337796
0.9  0.7605925414192195



![1638096639672](twitter_image/1638096639672.png)

| regparam | accuracy | difftime | difftime_m |
| -------- | -------- | -------- | ---------- |
| 0.1      | 0.763398 | 124653   | 2.07755    |
| 0.2      | 0.763686 | 276673   | 4.611217   |
| 0.3      | 0.763565 | 416267   | 6.937783   |
| 0.4      | 0.76345  | 556290   | 9.2715     |
| 0.5      | 0.763188 | 673753   | 11.22922   |
| 0.6      | 0.762892 | 782994   | 13.0499    |
| 0.7      | 0.762694 | 906857   | 15.11428   |
| 0.8      | 0.76255  | 1033613  | 17.22688   |
| 0.9      | 0.762471 | 1163341  | 19.38902   |

![1638097970247](twitter_image/1638097970247.png)

表情符号

Accuracy :  0.767326679304522

Accuracy :  0.7673232827496862

Accuracy :  0.7675560927831938

3.0               0.767389068607763

Accuracy :  0.7671166182492645



Accuracy :  0.7666956830008246
diffTime 82
emNum 69142

Accuracy :  0.7673131696324061

Accuracy :  0.7669982346524727
diffTime 88
emNum 69137

Accuracy :  0.7668218495272642
diffTime 94
emNum 7371

去掉amp后

Accuracy :  0.7668212351542333

Accuracy :  0.7668432754289888



Accuracy :  0.783614375
diffTime 79
emNum 7371

Accuracy :  0.783614375



Accuracy :  0.783987500000000

Accuracy :  0.7838681249999999

Accuracy :  0.784011875

Accuracy :  0.784011875

Accuracy :  0.7840543750000001



根据词云图的结果，添加停用词 

```
u
ll
amp
quot
```

<==******** Prediction accuracy compared to actual: 79.09894% ********==>

标准停用词

<==******** Prediction accuracy compared to actual: 79.08081% ********==>

没有停用词

<==******** Prediction accuracy compared to actual: 80.17431% ********==>

not good 替换成 notgood

<==******** Prediction accuracy compared to actual: 79.12606% ********==>

no money  替换 成 nomoney

<==******** Prediction accuracy compared to actual: 79.10381% ********==>

don't know 替换成  dontknow 

<==******** Prediction accuracy compared to actual: 79.12000% ********==>

can't see  替换成  cantsee 

<==******** Prediction accuracy compared to actual: 79.11600% ********==>

don't think  替换成  dontthink 

<==******** Prediction accuracy compared to actual: 79.10700% ********==>

全部替换  

<==******** Prediction accuracy compared to actual: 79.18106% ********==>

字符表情转为 字符串

<==******** Prediction accuracy compared to actual: 79.18075% ********==>



new4 stopwords, 79.09894
no stopwords , 80.17431
stopwords   , 79.08081
not good   , 79.12606 
no money   , 79.10381
don't know , 79.12000
can't see  , 79.12000
don't think ,79.10700
all not    , 79.18106
emoji      , 79.18075



![1638116195804](twitter_image/1638116195804.png)

accuracy  不是越高就越好，里面有许多虚假的数据

调整lambda参数

![1638115302229](twitter_image/1638115302229.png)