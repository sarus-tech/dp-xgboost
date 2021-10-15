package sarus 

import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import ml.dmlc.xgboost4j.scala.spark.TrackerConf

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

object SparkTraining {

  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      // scalastyle:off
      println("Usage: program input_path [cpu|gpu]")
      sys.exit(1)
    }

    val (treeMethod, numWorkers) = if (args.length == 2 && args(1) == "gpu") {
      ("gpu_hist", 1)
    } else ("auto", 2)

    val spark = SparkSession.builder().getOrCreate()
    val inputPath = args(0)

    val rawInput = spark.read
              .option("header", true)
              .option("inferSchema", true)
              .csv(inputPath)

    // compose all feature columns as vector
    val vectorAssembler = new VectorAssembler().
      setInputCols(Array("year", "mileage", "tax", "mpg")).
      setOutputCol("features")

    val xgbInput = vectorAssembler.transform(rawInput).select("features",
      "label")

    xgbInput.show 
    val Array(train, test) = xgbInput.randomSplit(Array(0.8, 0.2))

    val xgbParam = Map("eta" -> 0.5f,
      "max_depth" -> 5,
      "tracker_conf" -> TrackerConf(0, "python"),
      "objective" -> "reg:squarederror",
      "num_round" -> 10,
      "num_workers" -> 2,
      "tree_method" -> "approxDP",
      "dp_epsilon_per_tree" -> 10.0f,
      "min_child_weight" -> 50.0)

    val fmin = Array[Float](1996, 1, 0, 20)
    val fmax = Array[Float](2060, 170000, 580, 202)

    val xgbRegressor = new XGBoostRegressor(xgbParam).
      setFeaturesCol("features").
      setLabelCol("label").
      setFeatureBounds(fmin, fmax)

    println(xgbRegressor.fmin.mkString(",")) 
      
    val xgbRegModel = xgbRegressor.fit(train)
    val results = xgbRegModel.transform(train)
    results.show()
  }
}
