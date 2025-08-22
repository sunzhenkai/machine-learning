import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, greatest, lit, abs
from configparser import ConfigParser
from pyspark.sql.functions import sum as spark_sum

# 设置 JDK
os.environ["JAVA_HOME"] = "/lib/jvm/java-17-openjdk-amd64"

# 添加 jar 包
jars_dir = "/home/jovyan/jars"
jars_list = [
    os.path.join(jars_dir, f) for f in os.listdir(jars_dir) if f.endswith(".jar")
]
jars_str = ",".join(jars_list)
print(jars_str)

class MinIoConfig:
    def __init__(self, minio_config_file="~/.minioconfig"):
        config_path = os.path.expanduser(minio_config_file)
        parser = ConfigParser()
        parser.read(config_path)
        self.endpoint = parser.get("Credentials", "endpoint")
        self.access_key_id = parser.get("Credentials", "accessKeyID")
        self.access_key_secret = parser.get("Credentials", "accessKeySecret")

class OssConfig:
    def __init__(self, oss_config_file="~/.ossutilconfig"):
        config_path = os.path.expanduser(oss_config_file)
        parser = ConfigParser()
        parser.read(config_path)
        self.endpoint = parser.get("Credentials", "endpoint")
        self.access_key_id = parser.get("Credentials", "accessKeyID")
        self.access_key_secret = parser.get("Credentials", "accessKeySecret")

class SparkSessionBuilder:
    def __init__(self):
        self._spark_session = SparkSession.builder.appName("LocalPySparkExample")\
            .config("spark.jars", jars_str)\
            .master("local[*]")\
            .config("spark.driver.memory", "10g")\
            .config("spark.driver.maxResultSize", "4g")\
            .config("spark.sql.shuffle.partitions", "100")\
            .config("spark.default.parallelism", "100")\
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35")

    def with_minio(self):
        self.minio_config = MinIoConfig()
        self._spark_session.config("spark.hadoop.fs.s3a.endpoint", self.minio_config.endpoint)\
            .config("spark.hadoop.fs.s3a.access.key", self.minio_config.access_key_id)\
            .config("spark.hadoop.fs.s3a.secret.key", self.minio_config.access_key_secret)\
            .config("spark.hadoop.fs.s3a.path.style.access", "true")\
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        return self

    def with_oss(self):
        self.oss_config = OssConfig()
        self._spark_session.config("spark.hadoop.fs.oss.impl","org.apache.hadoop.fs.aliyun.oss.AliyunOSSFileSystem",)\
            .config("spark.hadoop.fs.oss.accessKeyId", self.oss_config.access_key_id)\
            .config("spark.hadoop.fs.oss.accessKeySecret", self.oss_config.access_key_secret)\
            .config("spark.hadoop.fs.oss.endpoint", self.oss_config.endpoint)
        return self

    def build(self):
        return self._spark_session.getOrCreate()