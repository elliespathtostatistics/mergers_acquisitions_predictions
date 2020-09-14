// Module load spark/2.4.0
// spark-shell --executor-memory 4G --driver-memory 32G --num-executors 64

import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
// dataset was created with nlp pipeline: stopword removal, lemmatization, n-grams, tfidf 
val corpus_df13 = spark.read.load("hdfs:/user/rpm295/bdad_final_project/tgt_100_10k")

def balanceDataset(dataset: DataFrame): DataFrame = {

    // Re-balancing (weighting) of records to be used in the logistic loss objective function
    val numNegatives = dataset.filter(dataset("wasAcquired") === 0).count
    val datasetSize = dataset.count
    val balancingRatio = (datasetSize - numNegatives).toDouble / datasetSize

    val calculateWeights = udf { d: Double =>
      if (d == 0.0) {
        1 * balancingRatio
      }
      else {
        (1 * (1.0 - balancingRatio))
      }
    }

    val weightedDataset = dataset.withColumn("classWeightCol", calculateWeights(dataset("wasAcquired")))
    weightedDataset
  }
// rebalance dataset
val corpus_df14 = balanceDataset(corpus_df13)
val tgt_split = corpus_df14.randomSplit(Array(0.8, 0.2), seed = 42)
val tgt_train = tgt_split(0)
val tgt_test = tgt_split(1)

val tgt_lr = new LogisticRegression()
	.setWeightCol("classWeightCol")
	.setLabelCol("wasAcquired")
	.setFeaturesCol("tfidf")
	.setElasticNetParam(.5)
	.setRegParam(.03)
// train lr model
val tgt_lrModel = tgt_lr.fit(tgt_train)
tgt_lrModel.setThreshold(.68)
val lr_res = tgt_lrModel.transform(tgt_test)
val lr_evaluator = new BinaryClassificationEvaluator().setLabelCol("wasAcquired")
// AuC
lr_evaluator.evaluate(lr_res)
// Other metrics
val b = lr_res.select("wasAcquired", "prediction")
val tp = b.where($"wasAcquired" === 1 && $"prediction" === 1).count
val tn = b.where($"wasAcquired" === 0 && $"prediction" === 0).count
val fp = b.where($"wasAcquired" === 0 && $"prediction" === 1).count
val fn = b.where($"wasAcquired" === 1 && $"prediction" === 0).count