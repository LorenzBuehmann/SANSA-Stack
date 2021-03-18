package net.sansa_stack.rdf.flink.qualityassessment.metrics.completeness

import org.apache.flink.api.scala.{DataSet, _}
import org.apache.jena.graph.Triple
import org.apache.jena.vocabulary.RDF

/**
  * @author Gezim Sejdiu
  */
object SchemaCompleteness {

  /**
    * This metric measures the ratio of the number of classes and relations
    * of the gold standard existing in g, and the number of classes and
    * relations in the gold standard.
    */
  def assessSchemaCompleteness(triples: DataSet[Triple]): Double = {

    /**
      * -->Rule->Filter-->
      * select (?p2, o) where ?s p=rdf:type isIRI(?o); ?p2 ?o2
      * -->Action-->
      * S+=?p2 && SC+=?o
      * -->Post-processing-->
      * |S| intersect |SC| / |SC|
      */

    val p2_o = triples.filter(f =>
      f.getPredicate.matches(RDF.`type`.asNode())
        && f.getObject.isURI())

    val S = p2_o.map(_.getPredicate).distinct(_.hashCode())
    val SC = triples.map(_.getObject).distinct(_.hashCode())

    val S_intersection_SC = S.collect().intersect(SC.collect()).distinct

    val SC_count = SC.count()
    val S_intersection_SC_count = S_intersection_SC.size

    if (SC_count > 0) S_intersection_SC_count.toDouble / SC_count.toDouble
    else 0.00
  }

}
