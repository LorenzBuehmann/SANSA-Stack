package net.sansa_stack.query.spark.api.impl


import java.util

import scala.collection.JavaConverters._

import org.aksw.jena_sparql_api.core.{QueryExecutionBaseSelect, QueryExecutionFactory, ResultSetCloseable}
import org.aksw.jena_sparql_api.utils.ResultSetUtils
import org.apache.jena.graph
import org.apache.jena.query.{Query, QueryExecution}
import org.apache.jena.sparql.core.Quad
import org.apache.jena.sparql.engine.binding.Binding
import org.apache.jena.sparql.modify.TemplateLib
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import net.sansa_stack.query.spark.api.domain.{QueryExecutionSpark, ResultSetSpark}

abstract class QueryExecutionSparkBase(query: Query, subFactory: QueryExecutionFactory, spark: SparkSession)
  extends QueryExecutionBaseSelect(query, subFactory)
  with QueryExecutionSpark {

  override def executeCoreSelect(query: Query): ResultSetCloseable = {
    val bindings = execSelectSpark().getBindings.collect().iterator

    val tmp = ResultSetUtils.create2(query.getProjectVars, bindings.asJava)

    new ResultSetCloseable(tmp)
  }

  override def execConstructTriples(): util.Iterator[graph.Triple] = {
    execConstructSpark().collect().iterator.asJava
  }

  override def executeCoreSelectX(query: Query): QueryExecution = null

  def execSelectSpark(): ResultSetSpark

  override def execConstructSpark(): RDD[org.apache.jena.graph.Triple] = {
    val template = query.getConstructTemplate
    val triples = template.getTriples

    execSelectSpark()
      .getBindings
      .mapPartitions((bindingIt: Iterator[Binding]) => TemplateLib.calcTriples(triples, bindingIt.asJava).asScala)
      .distinct
  }

  override def execConstructQuadsSpark(): RDD[Quad] = {
    val template = query.getConstructTemplate
    val quads = template.getQuads

    execSelectSpark()
      .getBindings
      .mapPartitions((bindingIt: Iterator[Binding]) => TemplateLib.calcQuads(quads, bindingIt.asJava).asScala)
      .distinct
  }
}
