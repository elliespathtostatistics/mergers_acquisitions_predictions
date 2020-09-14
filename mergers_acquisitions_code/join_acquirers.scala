// join csv of historical M&As by name onto csv of CIKs, we can then link then to 10k reports which are identified by CIK
import org.apache.spark.sql.SparkSession

val spark = SparkSession
  .builder()
  .appName("Spark SQL basic example")
  .config("spark.some.config.option", "some-value")
  .getOrCreate()

// For implicit conversions like converting RDDs to DataFrames
import spark.implicits._
// Load 2 csvs
val df0 = spark.read.format("csv").option("header", "true").load("bdad_final_project/master_set.csv")
val df1 = spark.read.format("csv").option("header", "true").load("bdad_final_project/cik.csv")

import org.apache.spark.sql.functions.to_date
// Convert announce date to date type
val dateFormat = "MM/dd/yy"
var df2 = df0.withColumn("Announce Date", to_date($"Announce Date", dateFormat)) 
// Drop unneccessary columns
val df3 = df2.drop("Deal Type", "Seller Name", "Announced Total Value (mil.)", "Payment Type", "Announced Total Value (mil.)", "Deal Status",  "Current Target SIC Code", "Current Acquirer SIC Code", "Current Seller SIC Code", "TV/EBITDA")
val df4 = df1.drop("Exchange", "Business", "Incorpotated", "IRS")
// Left join on Acquirer Name
val df5 = df4.join(df3, $"Name" === $"Acquirer Name", "leftouter")
/*
val df6 = df5.drop("Acquirer Name")
// Rename "Name" column to "Company Name" to avoid ambiguity during join
val df7 = df4.toDF("CIK", "Ticker", "Company Name")

val df8 = df6.join(df7, $"Target Name" === $"Company Name", "leftouter")
// Rename cols
val df9 = df8.drop("Company Name").toDF("CIK", "Ticker", "Name", "Announce Date", "Target Name", "Target CIK", "Target Ticker")
// Write to dfs
*/
df5.write.format("csv").save("hdfs:/user/rpm295/bdad_final_project/acq_merged")