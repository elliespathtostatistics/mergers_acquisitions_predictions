// requires Spark >= 2.0: spark2-shell
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

val stopWords = sc.textFile("hdfs:/user/rpm295/stopwords.txt")
val broadcastStopWords = sc.broadcast(stopWords.collect.toSet)

// read in 10-ks from hsds, then convert to lowercase
val corpus_rdd0 = sc.wholeTextFiles("hdfs:/user/rpm295/bdad_final_project/corpus/", minPartitions = 200)
val corpus_rdd1 = corpus_rdd0.map(t => (extractDateCik(t._1), t._2.toLowerCase))

// remove stop words
val corpus_rdd2 = corpus_rdd1.map(t => (t._1, t._2.split("\\W").filter(!broadcastStopWords.value.contains(_)).toSeq))

// remove numbers and empty strings, then convert back to strings
val corpus_rdd3 = corpus_rdd2.map(t => (t._1(0), t._1(1), t._2.map(w => w.replaceAll("[^a-z]", "")).filter(_.nonEmpty).mkString(" ")))

// convert from rdd to dataframe
val corpus_df0 = corpus_rdd3.toDF("CIK", "Report Date", "texts")

// perform tfidf, drop words that appear in < 100 reports
import org.apache.spark.ml.feature.Tokenizer

val tokenizer = new Tokenizer().setInputCol("texts").setOutputCol("tok_txt")
val corpus_df1 = tokenizer.transform(corpus_df0)

import org.apache.spark.ml.feature.NGram

val twogram = new NGram().setN(2).setInputCol("tok_txt").setOutputCol("two_grams")
val threegram = new NGram().setN(3).setInputCol("tok_txt").setOutputCol("three_grams")

val corpus_df2 = twogram.transform(corpus_df1)

val mergeArrays = udf((a: Seq[String], b: Seq[String]) => (a ++ b).toSet.toSeq)

val corpus_df3 = corpus_df2.withColumn("a", mergeArrays($"tok_txt", $"two_grams"))

val corpus_df4 = threegram.transform(corpus_df3)

val corpus_df5 = corpus_df4.withColumn("full", mergeArrays($"a", $"three_grams"))

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

val countvec_model: CountVectorizerModel = new CountVectorizer().setInputCol("full").setOutputCol("features").setMinDF(500).fit(corpus_df5)
val corpus_df6 = countvec_model.transform(corpus_df5)

val corpus_df7 = corpus_df6.withColumn("reportDate", to_date($"Report Date")).drop("Report Date")  

corpus_df7.write.save("hdfs:/user/rpm295/tgt_500_7")

import org.apache.spark.ml.feature.IDF
val idf = new IDF().setInputCol("features").setOutputCol("tfidf")
val idfModel = idf.fit(corpus_df7)
val corpus_df8 = idfModel.transform(corpus_df7)

corpus_df8.write.save("hdfs:/user/rpm295/tgt_500_8")
