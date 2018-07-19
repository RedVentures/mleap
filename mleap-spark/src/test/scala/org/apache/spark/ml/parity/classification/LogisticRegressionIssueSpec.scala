package org.apache.spark.ml.parity.classification

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.ml.parity.SparkParityBase
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructType}

class LogisticRegressionIssueSpec extends SparkParityBase {

  val inputsA = Seq(2.3, 4.1, 2.3, 2.2, 1.1, 11.0)
  val inputsB = Seq(2.3, 4.1, 2.3, 3.4, 3.2, 10.1)
  val inputsC = Seq(1.1, 2.2, 11.2, 2.2, 7.6, 5.6)
  val inputsD = Seq(0.1, 2.2, 4.2, 11.2, 3.4, 21.2)
  val inputsE = Seq(2.3, 4.1, 2.3, 3.2, 11.9, 43.2)
  val y = Seq(0, 1, 0, 0, 1, 0)
  val rows = spark.sparkContext.parallelize(Seq.tabulate(6) { i => Row(inputsA(i), inputsB(i), inputsC(i), inputsD(i), inputsE(i), y(i)) })
  val schema = new StructType().add("demo:a", DoubleType, nullable = false).add("demo:b", DoubleType, nullable = false).add("demo:c", DoubleType, nullable = false).add("demo:d", DoubleType, nullable = false).add("demo:e", DoubleType, nullable = false).add("y", IntegerType, nullable = false)

  val dataset = spark.sqlContext.createDataFrame(rows, schema)

  override val sparkTransformer: Transformer = new Pipeline().setStages(Array(
    new VectorAssembler().
      setInputCols(Array("demo:a", "demo:c", "demo:d")).
      setOutputCol("unscaled_continuous_features"),
    new StandardScaler().
      setInputCol("unscaled_continuous_features").
      setOutputCol("scaled_continuous_features"),
    new LogisticRegression().
      setFeaturesCol("scaled_continuous_features").
      setThreshold(0.05).
      setLabelCol("y").
      setPredictionCol("demo:prediction"))).fit(dataset)
}
