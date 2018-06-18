import pyspark as ps
import numpy as np
from pyspark import SparkContext
from pyspark.sql.functions import col, lower, regexp_replace, split
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec, Word2VecModel
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from nltk.stem.porter import *


def tokenize_df(df): 
    
    tokenizer = Tokenizer(inputCol="text", outputCol="vector")
    remover = StopWordsRemover()
    remover.setInputCol("vector")
    remover.setOutputCol("vector_no_stopw")
    stopwords = remover.getStopWords()
    stemmer = PorterStemmer()
    stemmer_udf = udf(lambda x: stem(x), ArrayType(StringType()))

    df = df.select(clean_text(col("text")).alias("text"))
    df = tokenizer.transform(df).select("vector")
    df = remover.transform(df).select("vector_no_stopw")
    df = (df
        .withColumn("vector_stemmed", stemmer_udf("vector_no_stopw"))
        .select("vector_stemmed")
        )
    
    return df

    
def clean_text(c):
  c = lower(c)
  c = regexp_replace(c, "^rt ", "")
  c = regexp_replace(c, "(https?\://)\S+", "")
  c = regexp_replace(c, "[^a-zA-Z0-9\\s]", "")
  #c = split(c, "\\s+") tokenization...
  return c


def stem(in_vec):
    out_vec = []
    for t in in_vec:
        t_stem = stemmer.stem(t)
        if len(t_stem) > 2:
            out_vec.append(t_stem)       
    return out_vec