package net.sansa_stack.ml.spark.kge

import scala.collection.JavaConverters._

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.nn.LookupTable
import com.intel.analytics.bigdl.nn.mkldnn.Sequential
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.jena.riot.{Lang, RDFDataMgr}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{Adam, Optimizer}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

/**
 * @author Lorenz Buehmann
 */
object Dummy2 {

  val k = 5

  def main(args: Array[String]): Unit = {
    Engine.init

    val triples = RDFDataMgr.loadModel("file:///tmp/rdf.nt", Lang.NTRIPLES)

    val entitiesToId = triples.getGraph.find().asScala.flatMap(t => Seq(t.getSubject, t.getObject)).toSet.zipWithIndex.map { case (entity, idx) => (entity, idx + 1) }.toMap
    val relationsToId = triples.getGraph.find().asScala.map(t => t.getPredicate).toSet.zipWithIndex.map { case (relation, idx) => (relation, idx + 1) }.toMap

    val encodedTriples = triples.getGraph.find().asScala.map(t => (entitiesToId(t.getSubject), relationsToId(t.getPredicate), entitiesToId(t.getObject))).toArray

    val array = encodedTriples.map(t => T(t._1.toFloat, t._2.toFloat, t._3.toFloat))

    val input = Tensor(T.array(array))
    println(s"input:\n$input")

    val numEntities = entitiesToId.size
    val numRelations = relationsToId.size
    val numTriples = encodedTriples.length

    val entityEmbeddings = LookupTable(numEntities, k).setWeightsBias(Array(initializeEmbeddingTensor(numEntities)))
    println(s"entity embeddings:\n${entityEmbeddings.weight}")

    val relationEmbeddings = LookupTable(numRelations, k).setWeightsBias(Array(initializeEmbeddingTensor(numRelations)))
    println(s"relation embeddings:\n${relationEmbeddings.weight}")

    val h = Select(2, 1)
    val r = Select(2, 2)
    val t = Select(2, 3)

    for (i <- 1 to 10) {
      val emb_h = entityEmbeddings.forward(h.forward(input))
      val emb_r = relationEmbeddings.forward(r.forward(input))
      val emb_t = entityEmbeddings.forward(t.forward(input))
//      val d = emb_h + emb_r - emb_t
      val add = CAddTable()
      val sub = CSubTable()
      val hr = add.forward(T(1 -> emb_h, 2 -> emb_r))
      val distance = sub.forward(T(1 -> hr, 2 -> emb_t))
      println(s"distance: $distance")

      val target = Tensor[Float](numTriples, 1)
      val criterion = L1Cost()
      val loss = criterion.forward(distance, target)
      println(s"loss: $loss")
      val gradients = criterion.backward(distance, target)
      println(s"gradients: $gradients")


      entityEmbeddings.backward(
        h.forward(input),
        h.backward(input,
          add.backward(
            T(1 -> emb_h, 2 -> emb_r),
            sub.backward(T(1 -> hr, 2 -> emb_t), gradients))))

      entityEmbeddings.backward(
        t.forward(input),
        t.backward(input,
          add.backward(
            T(1 -> emb_h, 2 -> emb_r),
            sub.backward(T(1 -> hr, 2 -> emb_t), gradients))))
    }

    println(entityEmbeddings.weight)

//    val trainSet = DataSet.array(Array(triples, target))
//
//    val input1 = Input()
//    val input2 = Input()
//    val cadd = CAddTable().inputs(input1, input2)
//    val graph = Graph(Array(input1, input2), cadd)
//
//    val graph2 = Graph(Array(input1, input2), cadd)
//
//    val model = Sequential().add(entityEmbeddings).add(Select(2, 1))
//    println(model.forward(input))
//
//
//    val optimizer = Optimizer(
//      model = graph,
//      dataset = trainSet,
//      criterion = L1Cost[Float]())
//
//    optimizer.optimize()



  }


  def initializeEmbeddingTensor(size: Int): Tensor[Float] = {
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
