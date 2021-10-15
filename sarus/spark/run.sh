#!/usr/bin/env zsh 

mvn clean package

spark-submit --jars ../../jvm-packages/xgboost4j/target/xgboost4j_2.12-1.5.0-SNAPSHOT.jar,../../jvm-packages/xgboost4j-spark/target/xgboost4j-spark_2.12-1.5.0-SNAPSHOT.jar \
    --class sarus.SparkTraining target/sarus-test-xgboost-java-1.0.jar \
    data/ford2.csv 