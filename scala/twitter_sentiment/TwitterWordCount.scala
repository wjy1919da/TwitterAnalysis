package scala.twitter_sentiment

import com.google.gson.Gson
import org.apache.spark.{SparkConf, SparkContext}

object TwitterWordCount {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("YingPing")
      .setMaster("local[1]")

    //核心创建SparkContext对象
    val sc = new SparkContext(conf)

    //WordCount
    sc.textFile("twitter/part-00000-fa75be32-63d1-4733-a427-d7dcfbbae557-c000.csv")
      .flatMap(_.split(" "))
      .map((_, 1))
      .reduceByKey(_ + _)
      //.repartition(1)
      .sortBy(_._2, false)
      .take(50)
      .map(x => {
        val map = new java.util.HashMap[String, String]()
        map.put("name", x._1)
        map.put("value", x._2 + "")
        map.put("itemStyle", "createRandomItemStyle()")
        map
      })
      .foreach(item => println(new Gson().toJson(item).replace("\"c", "c").replace(")\"", ")") + ","))
    // 借助http://echarts.baidu.com/echarts2/doc/example/wordCloud.html#infographic可以显示词云

    //停止SparkContext对象
    sc.stop()
  }

}
