lazy val sparkVersion = "2.1.0"
lazy val circeVersion = "0.6.1"
lazy val slf4jVersion = "1.7.21"

lazy val buildSettings = Seq(
  organization := "cs249",
  version := "1.0",
  scalaVersion := "2.11.8",
  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core",
    "org.apache.spark" %% "spark-mllib",
    "org.apache.spark" %% "spark-sql"
  ).map(_ % sparkVersion % "provided")
)


lazy val sf = project
  .settings(moduleName := "sf")
  .settings(buildSettings)
