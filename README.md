# Smart Home Energy Consumption Analysis using PySpark

## Overview

This project analyzes large-scale smart home energy consumption data using PySpark and Machine Learning techniques. The system processes smart meter and sensor data to identify energy usage patterns, perform exploratory data analysis, and forecast future household energy consumption.

The project demonstrates the use of Big Data Analytics for understanding energy demand, identifying peak consumption periods, and supporting efficient energy management.

---

## Objectives

* Process large-scale smart home energy datasets using PySpark.
* Perform data cleaning and preprocessing.
* Engineer time-based and environmental features.
* Analyze hourly, monthly, and seasonal energy consumption patterns.
* Build predictive machine learning models for energy forecasting.
* Generate visualizations to support decision-making.

---

## Technologies Used

* Python
* PySpark
* Apache Spark
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn

---

## Dataset Features

The dataset contains information such as:

* Home ID
* Appliance Type
* Energy Consumption (kWh)
* Date and Time
* Outdoor Temperature
* Season
* Household Size

---

## Methodology

### Data Preprocessing

* Data cleaning and null value handling
* Type conversion and schema detection
* Feature extraction from timestamps
* Missing value treatment using median imputation

### Feature Engineering

* Hour
* Day
* Month
* Year
* Temperature
* Household Size

### Machine Learning Models

1. Linear Regression
2. Random Forest Regressor

### Evaluation Metrics

* RMSE (Root Mean Square Error)
* MAE (Mean Absolute Error)
* R² Score

---

## Key Analysis Performed

* Hourly Energy Consumption Analysis
* Monthly Energy Consumption Trends
* Seasonal Energy Consumption Analysis
* Temperature vs Energy Correlation
* Appliance-wise Energy Usage Analysis
* Machine Learning-based Energy Forecasting

---

## Results

* Identified peak energy consumption during morning and evening hours.
* Observed the impact of temperature and seasonal changes on energy usage.
* Random Forest Regression provided better forecasting performance compared to Linear Regression.
* Generated visual insights for energy conservation and demand planning.

---

## Project Structure

Smart-Home-Energy-Consumption-Analysis/
│
├── smart_home_energy_consumption_large.csv
├── energy_analysis.py
├── requirements.txt
├── README.md
├── output/
│   ├── plots/
│   ├── predictions/
│   └── reports/
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python energy_analysis.py
```

---

## Applications

* Smart Home Monitoring
* Energy Consumption Forecasting
* Utility Load Planning
* Energy Conservation Systems
* Smart Grid Analytics

---

## Future Enhancements

* Real-time data streaming with Spark Streaming
* Deep Learning-based forecasting models
* Interactive dashboard using Power BI or Streamlit
* Smart Grid Integration

---

## Author

**Manasa K M**
