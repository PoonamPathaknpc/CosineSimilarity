
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.DistributedLDAModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.Row
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.catalyst.encoders.RowEncoder

import scala.collection.mutable


object MainApplication {


  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Usage: Part1 InputDir OutputDir")
    }

    val spark = SparkSession.builder.master("local").appName("Topic Modelling").getOrCreate()
    val sc = spark.sparkContext

    val sqlContext = spark.sqlContext

    // Extracting the data
    import sqlContext.implicits._
    var corpusdata = sqlContext.read.format("com.databricks.spark.csv").option("header", "true")
      .load(args(0))
      .select(regexp_replace($"title", "Äö|Äî|Äö|Äú|Ä√©|Äù|Äö", "").alias("title"),
        regexp_replace($"content", "Äö|Äî|Äö|Äú|Ä√©|Äù|Äö", "").alias("content"))

      .cache()

    corpusdata = corpusdata.filter($"content".isNotNull)

    var topics = sqlContext.read.format("com.databricks.spark.csv").option("header", "true")
      .option("delimiter", "[")
      .load(args(1)).cache()
    var cleantopic = topics.select(regexp_replace($"topic],", "[\\]]|\"", "").alias("topic"))
      .withColumn("topic", split($"topic", ",")).withColumn("id", monotonically_increasing_id())



    // CORPUSDATA PROCESSING
    //TFIDF of news as per topic column
    println("Logging: TFIDF of news as per topic column")

    val stopwords = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("tokens")

    val mltokenizer1 = new RegexTokenizer()
      .setInputCol("content")
      .setOutputCol("words")
      .setPattern("\\w+").setGaps(false)

    val pipeline1 = new Pipeline().setStages(Array(mltokenizer1, stopwords))
    var tfdata = pipeline1.fit(corpusdata).transform(corpusdata)


    var tfdatadf = tfdata.drop("content").drop("words")
    println("here")

    //tfdatadf.show(2, false)
    val DocumentNumber = 30 // have to be chnaged for the final execution

    // adding the topics as columns in corpus data
    println("Logging: adding the topics as columns in corpus data")
    cleantopic.select("topic", "id").rdd.collect.map(row1 => {

      val i = row1.getLong(1).toInt

      //Adding dummy columns for tf
      tfdatadf = tfdatadf.withColumn(s"topic_tf$i", lit(1.0))
      tfdatadf = tfdatadf.withColumn(s"topic$i", lit(row1.getAs[Seq[String]](0).mkString(",")))


      var idfs = Array(1.0, 1.0, 1.0, 1.0, 1.0) // to avoid division by zero situation
      var k = 0
      //println("tfdata")

      tfdatadf.rdd.collect()
        .foreach {
          row2 =>
            var arrayofTfs = Array(1.0, 1.0, 1.0, 1.0, 1.0)
            //println("topic1", row1.getAs[Seq[String]](0))
            row1.getAs[Seq[String]](0).flatMap {
              topicw => {
                val j = row1.getAs[Seq[String]](0).indexOf(topicw)
                val line = row2.getAs[Seq[String]](1).mkString(",")
                if (line.contains(topicw) && j != 5) {
                  idfs(j) = idfs(j) + 1
                  line.split(",").filter(w => (w == topicw)).groupBy(_.toString)
                    .map(ws => arrayofTfs(j) = (1 + Math.log(ws._2.size)))
                }
                else
                  Seq(1.0)
              }
            }
            val title = row2.getString(0)

            tfdatadf = tfdatadf.withColumn(s"topic_tf$i", when($"title" === title
              , lit(arrayofTfs.mkString(",")))
              .otherwise(col(s"topic_tf$i")))
        }

      //  idf calculation as per the documents
      for (k <- 0 to 4) {
        val a = (DocumentNumber + 1).toDouble / idfs(k)
        idfs(k) = Math.log(a)
      }
      // Adding dummy column for idf
      corpusdata = corpusdata.withColumn(s"topic_df$i", lit(1.0))
      tfdatadf = tfdatadf.withColumn(s"topic_idf$i", lit(idfs.mkString(",")))

    }
    )

    tfdatadf.show(2, false)
    println("Logging: converting tf-idf value to double arrays")

    val toDouble1 = udf((value: String) => value.split(",").map(_.toDouble))

    for (i <- 0 to 19) {
      tfdatadf = tfdatadf.withColumn(s"tfdouble$i", toDouble1($"topic_tf$i").alias(s"tfdouble$i"))
        .withColumn(s"idfdouble$i", toDouble1($"topic_idf$i").alias(s"idfdouble$i"))
        .drop($"topic_idf$i").drop($"topic_tf$i")

    }

    tfdatadf = tfdatadf.drop("tokens")
    tfdatadf.show(2, false)

    var columnNames = Seq.fill(61)("t")
    columnNames = columnNames.updated(0, "title")
    var k = 0
    for (i <- 0 to 57 by 3) {
      columnNames = columnNames.updated(i + 1, s"topic$k")
      columnNames = columnNames.updated(i + 2, s"tfdouble$k")
      columnNames = columnNames.updated(i + 3, s"idfdouble$k")
      k = k + 1
    }

    tfdatadf = tfdatadf.select(columnNames.head, columnNames.tail: _*)

    //tfdatadf.show(2, false)

    println("Logging: product t*idf")

    import org.apache.spark.sql.functions._
    import org.apache.spark.ml.linalg.Vectors
    def convertArrayToVector = udf((features: mutable.WrappedArray[Double])
    => Vectors.dense(features.toArray))

    var j = 0

    for (i <- 2 to 59 by 3) {
      var arrayofprods2 = Array(1.0, 1.0, 1.0, 1.0, 1.0)
      var title = ""

      // Adding dummy column for idf
      tfdatadf = tfdatadf.withColumn(s"topic_tfidf$j", lit(1.0))

      tfdatadf.rdd.collect()
        .foreach {

          row =>
            title = row.getString(0)
            //println(row.getAs[Seq[String]](0).mkString(","))
            val tfarray = row.getAs[Seq[Double]](i)
            val idfarray = row.getAs[Seq[Double]](i + 1)
            for (k <- 0 to 4) {
              arrayofprods2(k) = tfarray(k) * idfarray(k)
            }

            tfdatadf = tfdatadf.withColumn(s"topic_tfidf$j", when($"title" === title
              , lit(arrayofprods2.mkString(",")))
              .otherwise(col(s"topic_tfidf$j")))
        }
      j = j + 1

    }

    //tfdatadf.show(2, false)
    for (i <- 0 to 19) {
      tfdatadf = tfdatadf.withColumn(s"topic_tfidf$i", convertArrayToVector(toDouble1($"topic_tfidf$i")))
      tfdatadf = tfdatadf.withColumn(s"idfdouble$i", convertArrayToVector($"idfdouble$i"))

    }
    //
    //tfdatadf.show(2, false)

    // TOPIC DATA PROCESSING
    // TFIDF calculation of 20 topics from topic modelling

    println("Logging: Normalizing t*idf")
    for (i <- 0 to 19) {

      tfdatadf = tfdatadf.drop($"tfdouble$i")
      val normalizer = new Normalizer()
        .setInputCol(s"topic_tfidf$i")
        .setOutputCol(s"titlenormtfidf$i")

      val normalizer1 = new Normalizer()
        .setInputCol(s"idfdouble$i")
        .setOutputCol(s"topicnormtfidf$i")

      tfdatadf = normalizer.transform(tfdatadf)
      tfdatadf = normalizer1.transform(tfdatadf)


    }


    def vecToArray = udf((v: Vector) => v.toArray)

    for (i <- 0 to 19) {
      tfdatadf = tfdatadf.drop($"topic_tfidf$i")
      tfdatadf = tfdatadf.withColumn(s"titlenormtfidf$i", vecToArray($"titlenormtfidf$i"))
      tfdatadf = tfdatadf.withColumn(s"topicnormtfidf$i", vecToArray($"topicnormtfidf$i"))
      tfdatadf = tfdatadf.drop($"tfdouble$i").drop($"idfdouble$i")

    }


    //tfdatadf.show(2, false)

    var columnNames1 = Seq.fill(61)("t")


    columnNames1 = columnNames1.updated(0, "title")
    var t = 0
    for (i <- 0 to 57 by 3) {
      columnNames1 = columnNames1.updated(i + 1, s"topic$t")
      columnNames1 = columnNames1.updated(i + 2, s"topicnormtfidf$t")
      columnNames1 = columnNames1.updated(i + 3, s"titlenormtfidf$t")
      t = t + 1
    }


    tfdatadf = tfdatadf.select(columnNames1.head, columnNames1.tail: _*)

    //tfdatadf.show(2, false)


    println("Logging: get cosine vals and determine the topic")

    // Adding dummy column for cosine similarity value
    tfdatadf = tfdatadf.withColumn("most_similar_topic", lit("none"))


    var count = 0
    tfdatadf.rdd.collect()
      .foreach {
        row =>
          var topic = ""
          var title = row.getString(0)
          var cosinesimval = 0.0
          for (i <- 2 to 59 by 3) {

            val topicval = row.getAs[Seq[Double]](i)
            val titleval = row.getAs[Seq[Double]](i + 1)
            var sum = 0.0
            for (k <- 0 to 4) {
              sum = sum + (topicval(k) * titleval(k))
            }

            if (sum > cosinesimval) {
              cosinesimval = sum
              if (i == 2)
                topic = row.getString(1)
              else
                topic = row.getString(i + 2)
            }
          }

          // add the title to cosine val
          tfdatadf = tfdatadf.withColumn("most_similar_topic", when($"title" === title
            , lit(topic))
            .otherwise(col("most_similar_topic")))
          count = count + 1

      }

    // drop all the other columns
    for (i <- 0 to 19) {
      tfdatadf = tfdatadf.drop($"topic$i")
      tfdatadf = tfdatadf.drop($"topicnormtfidf$i")
      tfdatadf = tfdatadf.drop($"titlenormtfidf$i")
    }

    tfdatadf.show(5, false)
    tfdatadf.write.format("com.databricks.spark.csv").save(args(2))

  }

}


