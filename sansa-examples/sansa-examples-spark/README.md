# SANSA-Examples on Apache Spark
This is a SANSA-Examples repo for Apache Spark.

## Running the application on a Spark standalone cluster

To run the application on a standalone Spark cluster

1. Setup a Spark cluster
2. Build the application with Maven

  ```
  git clone https://github.com/SANSA-Stack/SANSA-Examples.git
  cd SANSA-Examples/sansa-examples-spark

  mvn clean package

  ```

3. Submit the application to the Spark cluster

  ```
  spark-submit \
		--class net.sansa_stack.examples.spark.<SANSA Layer>.<Example> \
		--master spark://spark-master:7077 \
 		/app/application.jar \
		SPARK_APPLICATION_ARGUMENTS  
  ```

## Running the application on a Spark standalone cluster via Spark Docker using BDE Platform

To run the SANSA-Examples application on BDE platform, execute the following commands:

```
  git clone https://github.com/SANSA-Stack/SANSA-Examples.git
  cd SANSA-Examples/sansa-examples-spark

  make --directory config/csswrapper/ hosts
  
  docker network create hadoop

  docker-compose up -d
```
Note:To make it run, you may need to modify your /etc/hosts file. There is a Makefile, which will do it automatically for you (you should clean up your /etc/hosts after demo).

After BDE platform is up and running, let’s throw some data into our HDFS now by using Hue FileBrowser runing in our network. To perform these actions navigate to 'hue' tab into http://demo.sansa-stack.local. Use “hue” username with any password to login into the FileBrowser (“hue” user is set up as a proxy user for HDFS, see hadoop.env for the configuration parameters). Click on “File Browser” in upper right corner of the screen and use GUI to create /user/root/input and /user/root/output folders and upload the data file into /input folder.
Go to HDFS tab into http://demo.sansa-stack.local and check if the file exists under the path ‘/user/root/input/yourfile’.

After we have all the configuration needed for our example, let’s run our sansa-examples.

```
docker build --rm=true -t sansa/sansa-examples-spark .
```
And then just run this image:
```
docker run --name sansa-examples-spark-app --net hadoop --link spark-master:spark-master \
-e ENABLE_INIT_DAEMON=false \
-d sansa/sansa-examples-spark

```

