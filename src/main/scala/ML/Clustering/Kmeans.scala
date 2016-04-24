package ML.Clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.types.{StructField, DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.functions.{udf, array, lit}

import scala.collection.mutable


object Kmeans {
  def main(args: Array[String]) {
    //Turning logging off so we can see the output
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    //Initialize the spark cluster in local mode
    val conf = new SparkConf().setMaster("local[4]").setAppName("Kmeans")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    // Create a schema for the dataframe since we are just reading a text file
    val customSchema = StructType(Array(
      StructField("C0", DoubleType, nullable = false),
      StructField("C1", DoubleType, nullable = false),
      StructField("C2", DoubleType, nullable = false)
    ))

    //Read the text file and apply the schema
    val data  = sqlContext
      .read
      .format("com.databricks.spark.csv")
      .option("delimiter", " ")
      .schema(customSchema)
      .load("src/main/resources/kmeans_data.txt")

    data.registerTempTable("data")

    //Create a user defined funciton that converts an array of doubles to an mllib vector type
    val vectorize =(s: Seq[Double]) => Vectors.dense(s.toArray)
    sqlContext.udf.register("vectorize", vectorize)

    //Create a new table by selecting all the current columns and vecotrizing c0, c1, c2
    val scrubbed = sqlContext.sql("Select *, vectorize(array(C0,C1,C2)) as features from data")
    scrubbed.show()
    scrubbed.printSchema()

    // Trains a k-means model
    val kmeans = new KMeans()
      .setK(2)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")

    val model = kmeans.fit(scrubbed)
    model.transform(scrubbed).show()

    // Shows the result
    println("Final Centers: ")
    model.clusterCenters.foreach(println)

    //Now the model can be applied to any table that has a field called features
    val contest = sqlContext.createDataFrame(Array(
      ("Larry", 9.0, 8.0, 7.0),
      ("Sally", 1.0, 5.0, 3.0),
      ("Brian", 1.0, 0.0, 0.0),
      ("Alex", 9.8, 4.0, 3.0),
      ("Kyle", 3.0, 3.0,3.0),
      ("Susie", 10.0, 10.0, 10.0),
      ("Bruce", 9.0, 8.0, 7.0)
    )).toDF("Contestant", "Swimming", "Running", "PoleVault")


    contest.registerTempTable("contest")
    val withVector = sqlContext.sql("Select *, vectorize(array(Swimming, Running, PoleVault)) as features from contest")

    //This is the application of the model to the withvecotr table
    model.transform(withVector).show()

  }
}
