package net.sansa_stack.rdf.flink.qualityassessment.metrics.relevancy

import org.apache.flink.api.scala._
import org.apache.jena.graph.Triple

/**
  * @author Gezim Sejdiu
  */
object CoverageDetail {

  /**
    * R2
    * This metric measures the the coverage (i.e. number of entities described
    * in a dataset) and level of detail (i.e. number of properties) in a dataset
    * to ensure that the data retrieved is appropriate for the task at hand.
    */
  def assessCoverageDetail(dataset: DataSet[Triple]): Double = {

    val triples = dataset.count().toDouble

    val predicates = dataset.map(_.getPredicate).distinct(_.hashCode()).count().toDouble

    val value = if (triples > 0.0) {
      predicates / triples
    } else 0

    value

  }
}
