
import org.apache.spark.sql.SparkSession
import co.theasi.plotly.{AxisOptions, LegendOptions, MarkerOptions, Plot, ScatterMode, ScatterOptions, XAnchor, YAnchor, draw, writer}

import org.apache.spark.sql.functions.{monotonically_increasing_id, regexp_replace, split}


import util.Random


object plot {

  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Usage: Part1 InputDir OutputDir")
    }

    implicit val server = new writer.Server {
      val credentials = writer.Credentials("poonamp", "puZvtsQZYAku4h9FqZtg")
      val url = "https://api.plot.ly/v2/"
    }




    val spark = SparkSession.builder.master("local").appName("Topic Modelling").getOrCreate()
    val sc = spark.sparkContext

    val sqlContext = spark.sqlContext

    import sqlContext.implicits._
    var trainData_withNoLabels = sqlContext.read.format("com.databricks.spark.csv").option("header", "true")
      .load(args(0)).cache()

    trainData_withNoLabels.show(2,false)

    var Topic_Labels = sqlContext.read.format("com.databricks.spark.csv").option("header", "true")
      .option("delimiter", "[")
      .load(args(1)).cache()
    var cleantopic_label = Topic_Labels.select(regexp_replace($"topic],", "[\\]]|\"", "").alias("topic"))
      .withColumn("topic", split($"topic", ","))


    var Topics = Seq.fill(20)("")
    var Topics_counts = Seq.fill(20)(1)


    var xs = Seq.fill(2)("")
    xs = xs.updated(0, "2016")
    xs = xs.updated(1, "2017")

    var xs1 = Seq.fill(2)(1.0)


    var p1 = Plot()


    var k = 0
    cleantopic_label.rdd.collect()
      .foreach {
        row =>
          println(row.getAs[Seq[String]](0).mkString(","))
          Topics = Topics.updated(k, row.getAs[Seq[String]](0).mkString(","))
          var Topics$k = Seq.fill(2)(1)
          var count = trainData_withNoLabels.filter($"Topic" === row.getAs[Seq[String]](0).mkString(",")).count()
          var count2016 = trainData_withNoLabels.filter($"Topic" === row.getAs[Seq[String]](0).mkString(",")
                          && $"Year" === "2016").count()
          var count2017 = trainData_withNoLabels.filter($"Topic" === row.getAs[Seq[String]](0).mkString(",")
            && $"Year" === "2017").count()
          //println(count2017)

          Topics$k = Topics$k.updated(0,1+ count2016.toInt)
          Topics$k = Topics$k.updated(1,1+ count2017.toInt)


          println(Topics$k)
          p1.withScatter(xs,Topics$k.map { _ + k + 5.0 } ,ScatterOptions().mode(ScatterMode.Marker).name("line+marker"))


          Topics_counts = Topics_counts.updated(k, count.toInt)

          k = k + 1
      }

    draw(p1, "scatter-mode")

    println(Topics_counts)

//    // Options common to both traces
//    val commonOptions = ScatterOptions().mode(ScatterMode.Marker)
//                       .marker(MarkerOptions().symbol("circle").lineWidth(1).size(16))
//
//    val p = Plot()
//      .withScatter(Topics_counts, Topics, commonOptions
//        .name("Percent of news related to topic")
//        .updatedMarker(_.color(156, 165, 196, 0.95).lineColor(156, 165, 196, 1.0)))
//      .xAxisOptions(
//        AxisOptions()// Plot axis options
//          .noGrid
//          .withLine
//          .lineColor(102, 102, 102)
//          .titleColor(204, 204, 204)
//          .tickFontColor(102, 102, 102)
//          .noAutoTick
//          .tickSpacing(10.0)
//          .tickColor(102, 102, 102))
//
//    // Add the plot to the figure
//    val figure = Figure()
//      .plot(p)
//      .title("News associated with Top 20 Headlines between 2016 and 2017")
//      .legend(LegendOptions()
//        .yAnchor(YAnchor.Middle)
//        .xAnchor(XAnchor.Right))
//      .leftMargin(100)
//      .rightMargin(40)
//      .bottomMargin(50)
//      .topMargin(80)
//      .width(900)
//      .height(600)
//      .paperBackgroundColor(254, 247, 234)
//      .plotBackgroundColor(254, 247, 234)
//
//
//    //draw(figure, "cosinesimilarity", writer.FileOptions(overwrite = true))
//    import org.json4s._
//    import org.json4s.native.JsonMethods._
//    draw(figure, "cosinesimilarity")
//

  }

}
