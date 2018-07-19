package org.apache.spark.ml.parity.classification

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.parity.SparkParityBase
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructType}

class MultinomialLogisticRegressionParitySpec extends SparkParityBase {

  val labels = Seq(0.0, 1.0, 2.0, 0.0, 1.0, 2.0)
  val ages = Seq(15, 30, 40, 50, 15, 80)
  val heights = Seq(175, 190, 155, 160, 170, 180)
  val weights = Seq(67, 100, 57, 56, 56, 88)

  val rows = spark.sparkContext.parallelize(Seq.tabulate(6) { i => Row(labels(i), ages(i), heights(i), weights(i)) })
  val schema = new StructType().add("label", DoubleType, nullable = false)
    .add("age", IntegerType, nullable = false)
    .add("height", IntegerType, nullable = false)
    .add("weight", IntegerType, nullable = false)

  override val dataset: DataFrame = spark.sqlContext.createDataFrame(rows, schema)

  override val sparkTransformer: Transformer = new Pipeline().setStages(Array(
    new VectorAssembler().
      setInputCols(Array("age", "height", "weight")).
      setOutputCol("features"),
    new LogisticRegression().setLabelCol("label"))).fit(dataset)
}
