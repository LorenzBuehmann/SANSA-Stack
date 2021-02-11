package net.sansa_stack.ml.spark.kge

import org.apache.jena.riot.{Lang, RDFDataMgr}
import scala.collection.JavaConverters._

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{File, Table}

/**
 * @author Lorenz Buehmann
 */
object Dummy {

  val k = 20

  def main(args: Array[String]): Unit = {
    import com.intel.analytics.bigdl.utils.T
    import com.intel.analytics.bigdl.tensor.Tensor
    import com.intel.analytics.bigdl.numeric.NumericFloat
    import com.intel.analytics.bigdl.dataset.DataSet


    val triples = RDFDataMgr.loadModel("file:///tmp/rdf.nt", Lang.TURTLE)

    val entitiesToId = triples.getGraph.find().asScala.flatMap(t => Seq(t.getSubject, t.getObject)).toSet.zipWithIndex.toMap
    val relationsToId = triples.getGraph.find().asScala.map(t => t.getPredicate).toSet.zipWithIndex.toMap

    val encodedTriples = triples.getGraph.find().asScala.map (t => (entitiesToId(t.getSubject), relationsToId(t.getPredicate), entitiesToId(t.getObject))).toArray

    val ne = entitiesToId.size
    val nr = relationsToId.size

    // entity embeddings
    var e = initialize(ne)
    // relation embeddings
    var r = normalize(initialize(nr))

    // get head and tail embeddings
    var h = e(encodedTriples.map {case (h, r, t) => h})
    var t = e(encodedTriples.map {case (h, r, t) => t})

//    score(h, r, t)
  }

  def initialize(size: Int): Tensor[Float] = {
    val s = Math.sqrt(k)
    Tensor[Float](size, k).rand(-6 / s, 6 / s)
  }

  def normalize(data: Tensor[Float]): Tensor[Float] = {
    data.div(data.abs().sum())
  }

  def L1(vec: Tensor[Float]): Float = {
    vec.abs().sum()
  }

  def L2(vec: Tensor[Float]): Float = {
    vec.pow(2).sqrt().sum()
  }

  def score(h: Tensor[Float], r: Tensor[Float], t: Tensor[Float]): Float = {
    -L1(h.add(r).sub(t))
  }



}
