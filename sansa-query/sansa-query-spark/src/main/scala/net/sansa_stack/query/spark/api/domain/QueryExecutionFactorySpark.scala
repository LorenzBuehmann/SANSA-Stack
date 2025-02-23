package net.sansa_stack.query.spark.api.domain

import org.aksw.jena_sparql_api.core.QueryExecutionFactory
import org.apache.jena.query.Query

/**
 * Place to make [[QueryExecutionSpark]] objects from [[Query]] objects or a string.
 *
 * @author Lorenz Buehmann
 */
trait QueryExecutionFactorySpark extends QueryExecutionFactory {
  override def createQueryExecution(query: Query): QueryExecutionSpark
  override def createQueryExecution(query: String): QueryExecutionSpark
}
