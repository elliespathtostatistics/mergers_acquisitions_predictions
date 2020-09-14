// spark-shell --executor-memory 16G --driver-memory 32G --num-executors 64 --packages databricks:spark-corenlp:0.4.0-spark2.4-scala2.11
import org.apache.spark.sql.SparkSession

val spark = SparkSession
  .builder()
  .appName("M&A Target Prediction")
  .getOrCreate()

// for implicit conversions like converting RDDs to DataFrames
import spark.implicits._
// function to extract data and CIK from filenames
def extractDateCik(s: String) = {
    val arr = s.split("/|_")
    val res = Array(arr(9), arr(11))
    res
}
import org.apache.spark.sql.functions._
import com.databricks.spark.corenlp.functions._
val version = "3.9.1"
val baseUrl = s"http://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp"
val model = s"stanford-corenlp-$version-models.jar" // 
val url = s"$baseUrl/$version/$model"
if (!sc.listJars().exists(jar => jar.contains(model))) {
  import scala.sys.process._
  // download model
  s"wget -N $url".!!
  // make model files available to driver
  s"jar xf $model".!!
  // add model to workers
  sc.addJar(model)
}

val stopWords = sc.textFile("hdfs:/user/rpm295/bdad_final_project/stopwords.txt")
val broadcastStopWords = sc.broadcast(stopWords.collect.toSet)

// read in 10-ks from hsds, then convert to lowercase
val corpus_rdd0 = sc.wholeTextFiles("hdfs:/user/rpm295/bdad_final_project/corpus/", minPartitions = 1000)
val corpus_rdd1 = corpus_rdd0.map(t => (extractDateCik(t._1), t._2.toLowerCase))

// remove stop words
val corpus_rdd2 = corpus_rdd1.map(t => (t._1, t._2.split("\\W").filter(!broadcastStopWords.value.contains(_)).toSeq))

// remove numbers and empty strings, then convert back to strings
val corpus_rdd3 = corpus_rdd2.map(t => (t._1(0), t._1(1), t._2.map(w => w.replaceAll("[^a-z]", "")).mkString(" "))).filter(t => t._3.nonEmpty)

// convert from rdd to dataframe
val corpus_df0 = corpus_rdd3.toDF("CIK", "Report Date", "texts")

//val lemmas = corpus_df1.withColumn("tok_txt", lemma('texts))
// perform tfidf, drop words that appear in < 100 reports
import org.apache.spark.ml.feature.Tokenizer

//val tokenizer = new Tokenizer().setInputCol("texts").setOutputCol("tok_txt")
val corpus_df1 = corpus_df0.withColumn("tok_txt", lemma('texts))

import org.apache.spark.ml.feature.NGram

val twogram = new NGram().setN(2).setInputCol("tok_txt").setOutputCol("two_grams")
val threegram = new NGram().setN(3).setInputCol("tok_txt").setOutputCol("three_grams")

val corpus_df2 = twogram.transform(corpus_df1)

val mergeArrays = udf((a: Seq[String], b: Seq[String]) => (a ++ b).toSet.toSeq)

val corpus_df3 = corpus_df2.withColumn("a", mergeArrays($"tok_txt", $"two_grams"))

val corpus_df4 = threegram.transform(corpus_df3)

val corpus_df5 = corpus_df4.withColumn("full", mergeArrays($"a", $"three_grams"))

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

val countvec_model: CountVectorizerModel = new CountVectorizer().setInputCol("full").setOutputCol("features").setMinDF(100).setMaxDF(20000).fit(corpus_df5)
val corpus_df7 = countvec_model.transform(corpus_df5).withColumn("reportDate", to_date($"Report Date")).drop("Report Date") 

//corpus_df7.write.save("hdfs:/user/rpm295/acq_500_7")

import org.apache.spark.ml.feature.IDF
val idf = new IDF().setInputCol("features").setOutputCol("tfidf")
val idfModel = idf.fit(corpus_df7)
val corpus_df8 = idfModel.transform(corpus_df7)

//corpus_df8.write.save("hdfs:/user/rpm295/output8_500")

// Load acquirers into df, drop everything but CIK col and date col
val acquirers_df0 = spark.read.format("csv").option("header", "false").load("bdad_final_project/acq_merged")
var acquirers_df1 = acquirers_df0.withColumn("Announce Date", to_date($"_c5")).drop("_c1", "_c4", "_c5", "_c6", "_c7")

// Join on CIK and calculate announceDate - releaseDate
acquirers_df1.createOrReplaceTempView("acquirers_df1")
corpus_df8.createOrReplaceTempView("corpus_df8")
val corpus_df9 = spark.sql("SELECT CIK, _c2 AS Name, `reportDate` AS reportDate, `Announce Date` AS announceDate, tfidf, _c3 AS SIC, DATEDIFF(acquirers_df1.`Announce Date`, corpus_df8.`reportDate`) AS diff FROM corpus_df8 JOIN acquirers_df1 ON corpus_df8.CIK = acquirers_df1._c0 ")
corpus_df9.createOrReplaceTempView("corpus_df9")

// if (0 < announceDate - releaseDate < 365) label wasAcquired as 1, else 0
val corpus_df10 = spark.sql("SELECT *, IF (diff < 365 AND diff >= 0, 1, 0) AS acquired FROM corpus_df9").distinct
corpus_df10.createOrReplaceTempView("corpus_df10")
val corpus_df11 = spark.sql("SELECT * FROM corpus_df10 WHERE acquired = 1").drop("announceDate", "tfidf", "SIC", "diff").distinct
corpus_df11.createOrReplaceTempView("corpus_df11")
val corpus_df12 = corpus_df9.drop("announceDate", "diff").distinct
corpus_df12.createOrReplaceTempView("corpus_df12")
val corpus_df13 = spark.sql("SELECT corpus_df12.CIK, corpus_df12.Name, corpus_df12.reportDate, tfidf, SIC, IF (acquired = 1, 1, 0) AS acquired FROM corpus_df12 LEFT JOIN corpus_df11 ON corpus_df12.CIK = corpus_df11.CIK and corpus_df12.reportDate = corpus_df11.reportDate")

corpus_df13.write.save("hdfs:/user/rpm295/acq_500_final_df_lemmas")

import org.apache.spark.ml.clustering.LDA
corpus_df13.createOrReplaceTempView("corpus_df13")
val corpus_df14 = spark.sql("SELECT * FROM corpus_df13 WHERE acquired = 1")
val lda = new LDA().setK(20).setMaxIter(20).setDocConcentration(0.25).setTopicConcentration(0.25).setFeaturesCol("tfidf")
val lda_model = lda.fit(corpus_df13)
val transformed = lda_model.transform(corpus_df13)
val topics = lda_model.describeTopics(5)

import scala.collection.mutable
val topic_word_idx = topics.select("termIndices").rdd.collect().map(_.getAs[mutable.WrappedArray[Int]](0).toArray)
topic_word_idx.zipWithIndex.foreach{case(arr, i) => {println("\n-----Topic " + i); arr.foreach(el =>
println(countvec_model.vocabulary(el)))} }

val x = 10

/*
module load spark/2.4.0
spark-shell
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
val countvec_model = CountVectorizerModel.load("bdad_final_project/countvec_model_500")
val corpus_df13 = spark.read.load("hdfs:/user/rpm295/acq_500_final_df")
*/