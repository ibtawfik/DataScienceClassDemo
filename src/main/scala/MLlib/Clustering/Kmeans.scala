package MLlib.Clustering

import org.apache.spark.mllib.clustering.{KMeansModel, KMeans}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.log4j.Logger
import org.apache.log4j.Level

object Kmeans {
  def main(args: Array[String]) {
    //Turning logging off so we can see the output
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    //Initialize the spark cluster in local mode
    val conf = new SparkConf().setMaster("local[4]").setAppName("Kmeans")
    val sc = new SparkContext(conf)

    //Load the date into the spark context.
    val data  = sc.textFile("src/main/resources/kmeans_data.txt")
    //This is now an rdd of strings. Each entry in the RDD is a row in the file.
    println("RDD OF STRINGS")
    data.foreach(println)

    //Create a dense vector out of the data
    //A quick aside on Scala syntax, in this case similar to Hadoop MR. Map is a function that works over a collection
    //and basically takes each input in the collection and maps(ie transforms) it to another value. This line says for
    //every string in the RDD split the string into an array of strings with the space being the delimiter, and use it to
    //create a dense vector.
    val vectorizedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble)))

    //We now have an RDD of vectors
    println("RDD OF VECTORS")
    vectorizedData.foreach(println)

    //Run the clustering algorithm
    val numberOfCentriods = 2
    val numberOfIterations = 20
    val clusters = KMeans.train(vectorizedData, numberOfCentriods, numberOfIterations)

    //Now that the model is trained we can do a few things:

    //Get the label for a new point
    val newPointPrediction = clusters.predict(Vectors.dense(Array(1.0,2.0,.05)))

    //Get the labels of an RDD of vectors as an RDD of Ints, beware this is not going to be in order, I'll show you later
    //how to get around that.
    val rddPredicitons = clusters.predict(vectorizedData)

    //Get the definition of the centriods
    val centriods = clusters.clusterCenters

    //There is a cost function available, this tells us how far points are from their nearest centriod. As a sum of the
    //squared errors
    val cost = clusters.computeCost(vectorizedData)

    //We can also save this model so that we can load it at runtime to classify new data points
    clusters.save(sc, "src/main/resources/KmeansModel")
    val sameClusters = KMeansModel.load(sc , "src/main/resources/KmeansModel")
    val sameClustersPointPrediciton = sameClusters.predict(Vectors.dense(Array(1.0,2.0,.05)))

    //Printing the results
    println("NEW POINT PREDICTION")
    println(newPointPrediction)

    println("SAME CLUSTER POINT PREDICTION")
    println(sameClustersPointPrediciton)

    println("CLUSTER CENTRIODS")
    centriods.foreach(println)

    println("CLUSTER ON RDD")
    //We are going to redo the clustering on a vector to create a vector to cluster pair
    val vectorToCluster = vectorizedData.map(v => (v , clusters.predict(v)))
    vectorToCluster.foreach(println)
  }
}
