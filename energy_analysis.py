from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofmonth, month, year, avg
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("SmartHomeEnergyAnalysis") \
    .getOrCreate()

# Load Dataset
df = spark.read.csv(
    "smart_home_energy_consumption_large.csv",
    header=True,
    inferSchema=True
)

# Data Cleaning
df_clean = (
    df.dropna()
      .withColumn("Home ID", col("Home ID").cast("int"))
      .withColumn("Energy Consumption (kWh)",
                  col("Energy Consumption (kWh)").cast("double"))
      .withColumn("Outdoor Temperature (°C)",
                  col("Outdoor Temperature (°C)").cast("double"))
      .withColumn("Household Size",
                  col("Household Size").cast("int"))
      .withColumn("Time", col("Time").cast("timestamp"))
)

# Feature Engineering
df_feat = (
    df_clean
    .withColumn("Hour", hour("Time"))
    .withColumn("Day", dayofmonth("Date"))
    .withColumn("Month", month("Date"))
    .withColumn("Year", year("Date"))
)

# Hourly Analysis
hourly = (
    df_feat.groupBy("Hour")
    .agg(avg("Energy Consumption (kWh)").alias("Average Energy"))
    .orderBy("Hour")
)

# Monthly Analysis
monthly = (
    df_feat.groupBy("Month")
    .agg(avg("Energy Consumption (kWh)").alias("Average Energy"))
    .orderBy("Month")
)

# Correlation
corr_temp = df_feat.corr(
    "Outdoor Temperature (°C)",
    "Energy Consumption (kWh)"
)

print("Temperature vs Energy Correlation:", corr_temp)

# Feature Vector
assembler = VectorAssembler(
    inputCols=[
        "Outdoor Temperature (°C)",
        "Household Size",
        "Hour",
        "Day",
        "Month"
    ],
    outputCol="features"
)

df_ml = assembler.transform(df_feat).select(
    "features",
    col("Energy Consumption (kWh)").alias("label")
)

# Train Test Split
train, test = df_ml.randomSplit([0.8, 0.2], seed=42)

# Linear Regression Model
lr = LinearRegression(
    featuresCol="features",
    labelCol="label"
)

lr_model = lr.fit(train)
lr_predictions = lr_model.transform(test)

# Random Forest Model
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="label",
    numTrees=100
)

rf_model = rf.fit(train)
rf_predictions = rf_model.transform(test)

# Evaluation
evaluator_rmse = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="r2"
)

print("\nLinear Regression Results")
print("RMSE:",
      evaluator_rmse.evaluate(lr_predictions))
print("R²:",
      evaluator_r2.evaluate(lr_predictions))

print("\nRandom Forest Results")
print("RMSE:",
      evaluator_rmse.evaluate(rf_predictions))
print("R²:",
      evaluator_r2.evaluate(rf_predictions))

# Convert to Pandas for Visualization
hourly_pd = hourly.toPandas()
monthly_pd = monthly.toPandas()

# Plot 1 - Energy by Hour
plt.figure(figsize=(8, 5))
plt.bar(hourly_pd["Hour"],
        hourly_pd["Average Energy"])
plt.title("Energy Consumption by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Energy (kWh)")
plt.savefig("energy_by_hour.png")
plt.show()

# Plot 2 - Energy by Month
plt.figure(figsize=(8, 5))
plt.plot(
    monthly_pd["Month"],
    monthly_pd["Average Energy"],
    marker="o"
)
plt.title("Energy Consumption by Month")
plt.xlabel("Month")
plt.ylabel("Average Energy (kWh)")
plt.savefig("energy_by_month.png")
plt.show()

spark.stop()
