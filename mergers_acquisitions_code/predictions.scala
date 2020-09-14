// spark-shell --executor-memory 16G --driver-memory 32G --num-executors 64 --packages databricks:spark-corenlp:0.4.0-spark2.4-scala2.11
import org.apache.spark.ml.feature.{StringIndexerModel, VectorAssembler}
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
ldaimport org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import com.databricks.spark.corenlp.functions._
import spark.implicits._
import scala.sys.process._
import scala.collection.mutable

val acq0 = spark.read.load("bdad_final_project/predictions_2019_acquirer")
val tgt0 = spark.read.load("bdad_final_project/tgt_predictions_2019")
val cik = spark.read.format("csv").option("header", "true").load("bdad_final_project/cik.csv")

val acq = acq0.filter(acq0("prediction") === 1)
val tgt = tgt0.filter(tgt0("prediction") === 1)
val y = acq.withColumn("SIC1", (acq("SIC") / 10).cast("integer"))
val z = tgt.withColumn("SIC1", (tgt("SIC") / 10).cast("integer"))
val acq = y
val tgt = z

acq.createOrReplaceTempView("acq")
tgt.createOrReplaceTempView("tgt")
cik.createOrReplaceTempView("cik")

val ap0 = spark.sql("SELECT acq.CIK, acq.Name, cik.Ticker, acq.SIC, acq.SIC1, probability FROM acq JOIN cik ON acq.CIK = cik.CIK")
val tp0 = spark.sql("SELECT tgt.CIK, tgt.Name, cik.Ticker, tgt.SIC, tgt.SIC1, probability FROM tgt JOIN cik ON tgt.CIK = cik.CIK")
val second = udf((v: org.apache.spark.ml.linalg.Vector) => v.toArray(1))

val ap = ap0.withColumn("prob", second($"probability")).drop("probability")
val tp = tp0.withColumn("prob", second($"probability")).drop("probability")

ap.createOrReplaceTempView("ap")
tp.createOrReplaceTempView("tp")

val ma = spark.sql("SELECT ap.Name AS acquirerName, tp.Name AS targetName, ap.CIK AS acqCIK, tp.CIK AS tgtCIK, ap.Ticker AS acqTicker, tp.Ticker AS tgtTicker, ap.SIC AS acqSIC, tp.SIC AS tgtSIC, ap.prob * tp.prob AS prob FROM ap JOIN tp ON ap.SIC1 = tp.SIC1 AND ap.CIK != tp.CIK") 

ma.rdd.coalesce(1).map(s => s.toString.drop(1).dropRight(1)).saveAsTextFile("ma_FINAL")
tp.rdd.coalesce(1).map(s => s.toString.drop(1).dropRight(1)).saveAsTextFile("tp_FINAL")
ap.rdd.coalesce(1).map(s => s.toString.drop(1).dropRight(1)).saveAsTextFile("ap_FINAL")



