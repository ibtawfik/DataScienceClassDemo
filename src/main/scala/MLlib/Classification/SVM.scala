package MLlib.Classification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}


object SVM {
  def main(args: Array[String]) {
    //Turning logging off so we can see the output
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    //Initialize the spark cluster in local mode
    val conf = new SparkConf().setMaster("local[4]").setAppName("Kmeans")
    val sc = new SparkContext(conf)

    //This dataset is in libsvm format, it's just a format that has the category label followed by a sparse vector that
    //is represented as index:value
    //This data is a labeled point object that has a label and features
    val data = MLUtils.loadLibSVMFile(sc, "src/main/resources/sample_libsvm_data.txt")

    //Splitting the data into a training and test set randomly
    val split = data.randomSplit(Array(.6,.4))
    val training = split(0)
    val test = split(1)

    //train the model
    val model = SVMWithSGD.train(training, 100)

    //Same as the the kmeans example we get a label using the predict function
    val predicition = test.map(p => (p.label, model.predict(p.features)))
    predicition.foreach(println)
  }
}
