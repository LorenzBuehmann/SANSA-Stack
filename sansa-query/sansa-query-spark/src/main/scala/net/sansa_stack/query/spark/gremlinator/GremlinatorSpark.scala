package net.sansa_stack.query.spark.gremlinator

import net.sansa_stack.query.spark.gremlinator.sparql2gremlin.QuaryTranslator
import net.sansa_stack.query.spark.gremlinator.utils.Config._
import org.apache.spark.sql.SparkSession
import org.apache.tinkerpop.gremlin.process.computer.traversal.TraversalVertexProgram
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource
import org.apache.tinkerpop.gremlin.spark.process.computer.SparkGraphComputer
import org.apache.tinkerpop.gremlin.spark.structure.Spark
import org.apache.tinkerpop.gremlin.structure.Graph
import org.apache.tinkerpop.gremlin.structure.util.GraphFactory

/*
   * GremlinatorSpark - a SPARQL to gremlin query rewriter on Spark.
 */

object GremlinatorSpark {

  @transient var spark: SparkSession = _

  def apply(sparql: String): GraphTraversalSource = {

    val gremlinSpark = Spark.create(spark.sparkContext)
    val sparkComputerConnection = GraphFactory.open(getSparkConfig) // spark.conf.getAll.toString())

    val gremlinquery = QuaryTranslator(sparql, sparkComputerConnection)
    run(sparkComputerConnection, gremlinquery.toString())
  }

  def run(sparkComputerConnection: Graph, gremlinQuery: String, lang: String = "gremlin-groovy"): GraphTraversalSource = {

    val sparkGraphComputer = getSparkGraphComputer(sparkComputerConnection)

    val traversalSource = sparkGraphComputer.program(TraversalVertexProgram.build()
      .traversal(
        sparkComputerConnection.traversal().withComputer(classOf[SparkGraphComputer]),
        lang,
        gremlinQuery)
      .create(sparkComputerConnection))
      .submit()
      .get()

    traversalSource.graph().traversal()
    // traversalSource.graph().compute().submit().get
  }

}
