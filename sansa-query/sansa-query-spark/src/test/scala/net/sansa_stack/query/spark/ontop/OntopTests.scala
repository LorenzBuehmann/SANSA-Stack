package net.sansa_stack.query.spark.ontop

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import net.sansa_stack.query.spark.api.domain.QueryExecutionFactorySpark
import net.sansa_stack.query.tests.util.ResultSetCompareUtils
import net.sansa_stack.rdf.spark.io._
import org.apache.jena.query.{Query, QueryFactory, ResultSet, ResultSetFactory}
import org.apache.jena.riot.Lang
import org.apache.jena.sparql.resultset.ResultSetCompare
import org.apache.spark.SparkConf
import org.scalatest.FunSuite

import java.io.{File, FileInputStream}
import scala.io.Source

class OntopTests extends FunSuite with DataFrameSuiteBase {

  var qef: QueryExecutionFactorySpark = _

  override def beforeAll(): Unit = {
    super.beforeAll()

    val input = getClass.getResource("/sansa-sparql-ts/bsbm/bsbm-sample.nt").getPath

    val triples = spark.rdf(Lang.NTRIPLES)(input)

    qef = new QueryEngineFactoryOntop(spark).create(triples)
  }

  override def conf(): SparkConf = {
    val conf = super.conf
    conf
      .set("spark.sql.crossJoin.enabled", "true")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "net.sansa_stack.rdf.spark.io.JenaKryoRegistrator")
    conf
  }

  val queries = List("q1", "q2", "q3")

  queries.foreach(q => {
    test(s"Test Ontop with BSBM $q") {
      val src = Source.fromFile(getClass.getResource(s"/sansa-sparql-ts/bsbm/bsbm-$q.rq").getPath)
      val queryString = src.getLines.mkString("\n")
      src.close()
      val query = QueryFactory.create(queryString)

      val rs = qef.createQueryExecution(query).execSelect()

      val rsTarget = ResultSetFactory.fromXML(new FileInputStream(new File(getClass.getResource(s"/sansa-sparql-ts/bsbm/bsbm-$q.srx").getPath)))

      assert(ResultSetCompareUtils.resultSetEquivalent(query, rs, rsTarget))
    }
  })
}
