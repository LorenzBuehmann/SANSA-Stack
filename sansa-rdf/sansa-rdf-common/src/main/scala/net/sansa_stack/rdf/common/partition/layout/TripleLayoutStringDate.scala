package net.sansa_stack.rdf.common.partition.layout

import java.time.LocalDate
import java.time.format.DateTimeFormatter

import scala.reflect.runtime.universe.{Type, typeOf}

import org.apache.jena.graph.Triple

import net.sansa_stack.rdf.common.partition.core.RdfPartitionerDefault
import net.sansa_stack.rdf.common.partition.schema.SchemaStringDate



/**
 * Triple layout for s[String], o[Date] columns.
 *
 * @author Lorenz Buehmann
 */
object TripleLayoutStringDate
  extends TripleLayout {
  override def schema: Type = typeOf[SchemaStringDate]

  // this is a more error tolerant date parser also allowing 1999-5-5 etc.
  private val formatter = DateTimeFormatter.ofPattern("u-M-d")

  override def fromTriple(t: Triple): SchemaStringDate = {
    val s = t.getSubject
    val o = t.getObject
    val v = if (o.isLiteral) {
      java.sql.Date.valueOf(LocalDate.parse(o.getLiteralLexicalForm, formatter))
    } else throw new RuntimeException(s"Layout only for date literals: $t")

    val sStr = RdfPartitionerDefault.getUriOrBNodeString(s)

    SchemaStringDate(sStr, v)
  }
}
