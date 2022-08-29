# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/test.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "test_csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `test_csv`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "test_csv"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

file_location = "/FileStore/tables/train.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
np.random.seed(60)

# COMMAND ----------

from pyspark.sql.functions import col, lower
data = df.select(lower(col('Category')),lower(col('Descript')))\
        .withColumnRenamed('lower(Category)','Category')\
        .withColumnRenamed('lower(Descript)', 'Description')
data.cache()
print('Dataframe Structure')
print('----------------------------------')
print(data.printSchema())
print(' ')
print('Dataframe preview')
print(data.show(5))
print(' ')
print('----------------------------------')
print('Total number of rows', df.count())

# COMMAND ----------

def top_n_list(df,var, N):
    
    #To determine the top N numbers of the list
    
    print("Total number of unique value of"+' '+var+''+':'+' '+str(df.select(var).distinct().count()))
    print(' ')
    print('Top'+' '+str(N)+' '+'Crime'+' '+var)
    df.groupBy(var).count().withColumnRenamed('count','totalValue')\
    .orderBy(col('totalValue').desc()).show(N)
    
    
top_n_list(data, 'Category',10)
print(' ')
print(' ')
top_n_list(data,'Description',10)

# COMMAND ----------

data.select('Category').distinct().count()

# COMMAND ----------

training, test = data.randomSplit([0.7,0.3], seed=60)

print("Training Dataset Count:", training.count())
print("Test Dataset Count:", test.count())

# COMMAND ----------



# COMMAND ----------


from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, OneHotEncoder, StringIndexer, VectorAssembler, HashingTF, IDF, Word2Vec
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes 
from pyspark.ml.classification import RandomForestClassifier

# COMMAND ----------

regex_tokenizer = RegexTokenizer(pattern='\\W')\
                  .setInputCol("Description")\
                  .setOutputCol("tokens")

# COMMAND ----------

# 2 Stopwords remover
extra_stopwords = ['http','amp','rt','t','c','the']
stopwords_remover = StopWordsRemover()\
                    .setInputCol('tokens')\
                    .setOutputCol('filtered_words')\
                    .setStopWords(extra_stopwords)

# COMMAND ----------

# 3 Countvectorizer
count_vectors = CountVectorizer(vocabSize=10000, minDF=5)\
               .setInputCol("filtered_words")\
               .setOutputCol("features")

# COMMAND ----------


hashingTf = HashingTF(numFeatures=10000)\
            .setInputCol("filtered_words")\
            .setOutputCol("raw_features")

# COMMAND ----------

idf = IDF(minDocFreq=5)\
        .setInputCol("raw_features")\
        .setOutputCol("features")

# COMMAND ----------

word2Vec = Word2Vec(vectorSize=1000, minCount=0)\
           .setInputCol("filtered_words")\
           .setOutputCol("features")

# COMMAND ----------

label_string_idx = StringIndexer()\
                  .setInputCol("Category")\
                  .setOutputCol("label")

# COMMAND ----------

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)

nb = NaiveBayes(smoothing=1)

rf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees = 100, \
                            maxDepth = 4, \
                            maxBins = 32)

# COMMAND ----------

# Logistic Regression using Count_Vectors
pipeline_cv_lr = Pipeline().setStages([regex_tokenizer,stopwords_remover,count_vectors,label_string_idx, lr])
model_cv_lr = pipeline_cv_lr.fit(training)
predictions_cv_lr = model_cv_lr.transform(test)

# COMMAND ----------

print('Top 5 predictions')
print(' ')
predictions_cv_lr.select('Description','Category',"probability","label","prediction")\
                                        .orderBy("probability", ascending=False)\
                                        .show(n=5, truncate=30)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator 
evaluator_cv_lr = MulticlassClassificationEvaluator().setPredictionCol("prediction").evaluate(predictions_cv_lr)
print(' ')
print('Accuracy')
print(' ')
print('accuracy:{}:'.format(evaluator_cv_lr))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# Random forest Using TF-IDF Features

pipeline_idf_rf = Pipeline().setStages([regex_tokenizer,stopwords_remover,hashingTf, idf, label_string_idx, rf])

model_idf_rf = pipeline_idf_rf.fit(training)
predictions_idf_rf = model_idf_rf.transform(test)
# predictions.filter(predictions['prediction'] == 0) \
#     .select("Descript","Category","probability","label","prediction") \
#     .orderBy("probability", ascending=False) \
#     .show(n = 10, truncate = 30

# COMMAND ----------

evaluator_idf_rf = MulticlassClassificationEvaluator().setPredictionCol("prediction").evaluate(predictions_idf_rf)
print(' ')
print('Accuracy')
print(' ')
print('                          accuracy:{}:'.format(evaluator_idf_rf))

# COMMAND ----------




# COMMAND ----------

# Logistic Regression using TF-IDF features
pipeline_idf_lr = Pipeline().setStages([regex_tokenizer,stopwords_remover,hashingTf, idf, label_string_idx, lr])
model_idf_lr = pipeline_idf_lr.fit(training)
predictions_idf_lr = model_idf_lr.transform(test)

# COMMAND ----------

# print('check Top 5 predictions')
# print(' ')
# predictions_idf_lr.select('Description','Category',"probability","label","prediction")\
#                                         .orderBy("probability", ascending=False)\
#                                         .show(n=5, truncate=30)

# COMMAND ----------

evaluator_idf_lr = MulticlassClassificationEvaluator().setPredictionCol("prediction").evaluate(predictions_idf_lr)
print(' ')
print('Accuracy')
print(' ')
print('Accuracy:{}:'.format(evaluator_idf_lr))

# COMMAND ----------

# Naivw bayes using TF-IDF Features
pipeline_idf_nb = Pipeline().setStages([regex_tokenizer,stopwords_remover,hashingTf, idf, label_string_idx, nb])
model_idf_nb = pipeline_idf_nb.fit(training)
predictions_idf_nb = model_idf_nb.transform(test)

evaluator_idf_nb = MulticlassClassificationEvaluator().setPredictionCol("prediction").evaluate(predictions_idf_nb)
print(' ')
print('Accuracy')
print(' ')
print('accuracy:{}:'.format(evaluator_idf_nb))

# COMMAND ----------

# Random Forest 

pipeline_cv_rf = Pipeline().setStages([regex_tokenizer,stopwords_remover,count_vectors,label_string_idx, rf])
model_cv_rf = pipeline_cv_rf.fit(training)
predictions_cv_rf = model_cv_rf.transform(test)

evaluator_cv_rf = MulticlassClassificationEvaluator().setPredictionCol("prediction").evaluate(predictions_cv_rf)
print(' ')
print('Accuracy')
print(' ')
print('accuracy:{}:'.format(evaluator_cv_rf))

# COMMAND ----------

pipeline_cv_nb = Pipeline().setStages([regex_tokenizer,stopwords_remover,count_vectors,label_string_idx, nb])
model_cv_nb = pipeline_cv_nb.fit(training)
predictions_cv_nb = model_cv_nb.transform(test)

evaluator_cv_nb = MulticlassClassificationEvaluator().setPredictionCol("prediction").evaluate(predictions_cv_nb)
print(' ')
print('Accuracy')
print(' ')
print('accuracy:{}:'.format(evaluator_cv_nb))

# COMMAND ----------



# COMMAND ----------


