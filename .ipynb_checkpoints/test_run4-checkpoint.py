from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, FloatType
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, PCA
import pyspark.sql.functions as F
from pyspark.ml.image import ImageSchema
from pyspark.ml.linalg import DenseVector, VectorUDT
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import sys
import pickle
import os
import boto3

# aws emr create-cluster --name test-emr-cluster3 --release-label emr-5.28.0 \
#  --instance-count 3 --instance-type m5.xlarge --applications Name=JupyterHub Name=Spark Name=Hadoop \
#   --ec2-attributes SubnetIds=subnet-04df75983a93c5bec,KeyName=Zephyrus --bootstrap-actions Path="s3://ocp8/emr_bootstrap3.sh"\
#    --region eu-west-3 --log-uri s3://ocp8/clusterlogs


model = VGG16(weights="imagenet", include_top=False)


def feature_extraction(x):
    img = image.load_img(x, target_size=(100, 100, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features = features.reshape(3, 3, 512)
    return features


def image_to_array(path):
    im = plt.imread(path)
    data = np.asarray(im)
    data = data.flatten()
    return data


def to_list(v):
    return v.toArray().tolist()


def split_array_to_list(col):
    return F.udf(to_list, ArrayType(FloatType()))(col)


spark = (
    SparkSession.builder.master("local").appName("SparkByExamples.com").getOrCreate()
)

spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", "AKIA3FBVFAGLF7WW6BWD")
spark._jsc.hadoopConfiguration().set(
    "fs.s3a.secret.key", "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"
)
spark._jsc.hadoopConfiguration().set(
    "fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
)
spark._jsc.hadoopConfiguration().set("com.amazonaws.services.s3.enableV4", "true")
spark._jsc.hadoopConfiguration().set(
    "fs.s3a.aws.credentials.provider",
    "org.apache.hadoop.fs.s3a.BasicAWSCredentialsProvider",
)
spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "eu-west-3.amazonaws.com")


print("Launch App..")
if __name__ == "__main__":
    print("Initiating main..")

    # image_folder = "file:///home/hysterio/code/open-classrooms/data/fruits/fruits-360_dataset/fruits-360/Training/"
    # types = glob(image_folder + "Apricot/*.jpg")
    # print(types)
    # list_pictures = types
    # types = glob(image_folder + "*/")
    # types = [a[87:-1] for a in types]

    print("Get files")
    # data="/home/hysterio/code/open-classrooms/data/fruits/fruits-360_dataset/fruits-360/Training/Apricot/*"
    data_url = "s3://ocp8/fruits-360_dataset/fruits-360/Training/Apricot/*"
    df = spark.read.format("image").load(data_url)
    ImageSchema.imageFields

    feat = F.udf(lambda x: DenseVector(feature_extraction(x)), VectorUDT())

    df = df.withColumn("vecs", feat("image.origin"))
    data = df.select("vecs")

    scaler = MinMaxScaler(inputCol="vecs", outputCol="scaledvecs")
    scaledData = scaler.fit(data).transform(data)
    # scaledData.write.parquet("s3://ocp8/output.parquet", mode="overwrite")
    pca = PCA(k=10, inputCol="scaledvecs", outputCol="pcavecs")
    model = pca.fit(scaledData)
    reduced_data = model.transform(scaledData).select("pcavecs")

    reduced_data.write.parquet("s3://ocp8/output.parquet", mode="overwrite")

    # s3 = boto3.resource(
    #     's3',
    #     aws_access_key_id = 'key_id',
    #     aws_secret_access_key = 'aws_secret_access_key',
    #     region_name = 'eu-west-3'
    # )
    # csv_buffer = StringIO()
    # reduced_data.to_csv(csv_buffer)
    # s3.Object('ocp8', 'pca_results.csv').put(Body=csv_buffer.getvalue())

    # s3 = boto3.resource("s3")
    # s3.Object("ocp8", "DONE.txt").put(Body=bytes(data))

    # s3 = boto3.resource("s3")
    # object = s3.Object("ocp8", "pickled_object.pkl")
    # pickle_obj = pickle.dumps(new_data)
    #
    #    try:
    #        object.put(Body=pickle_obj)
    #    except:
    #        print("Error in writing to S3 bucket")

    spark.stop()
