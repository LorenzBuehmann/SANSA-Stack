package net.sansa_stack.query.spark.graph.jena

trait Ops extends Serializable {

  def getTag: String

  def execute(): Unit

  def getId: Int
}
