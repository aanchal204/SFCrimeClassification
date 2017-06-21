# SFCrimeClassification
The project is in Spark scala. To run:
Go to the folder which contains build.sbt file.
Run: sbt "project <folder name>" clean assembly
Go to ‘folder’ and run the code using spark-submit. (You need to pass the path to the train.csv as a command line argument)
 
The submitted folder contains the .scala code and the build.sbt file. 
We have also provided the dataset file train.csv which is used for the classification.
