package net.sansa_stack.inference.rules.plan

import java.util.Properties

import org.apache.calcite.config.{CalciteConnectionConfig, CalciteConnectionConfigImpl, CalciteConnectionProperty}
import org.apache.calcite.jdbc.CalciteSchema
import org.apache.calcite.plan.{Context, RelOptCluster, RelOptPlanner, RelOptSchema}
import org.apache.calcite.prepare.CalciteCatalogReader
import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.schema.SchemaPlus
import org.apache.calcite.sql.parser.SqlParser
import org.apache.calcite.tools.Frameworks.PlannerAction
import org.apache.calcite.tools.{FrameworkConfig, Frameworks, RelBuilder}

/**
  * @author Lorenz Buehmann
  */
class SimpleRelBuilder (
                         context: Context,
                         cluster: RelOptCluster,
                         relOptSchema: RelOptSchema)
  extends RelBuilder(
    context,
    cluster,
    relOptSchema) {

  def getPlanner: RelOptPlanner = cluster.getPlanner

  def getCluster: RelOptCluster = cluster
}

object SimpleRelBuilder {

  def create(config: FrameworkConfig): SimpleRelBuilder = {
    // prepare planner and collect context instances
    val clusters: Array[RelOptCluster] = Array(null)
    val relOptSchemas: Array[RelOptSchema] = Array(null)
    val rootSchemas: Array[SchemaPlus] = Array(null)
    Frameworks.withPlanner(new PlannerAction[Void] {
      override def apply(
                          cluster: RelOptCluster,
                          relOptSchema: RelOptSchema,
                          rootSchema: SchemaPlus)
      : Void = {
        clusters(0) = cluster
        relOptSchemas(0) = relOptSchema
        rootSchemas(0) = rootSchema
        null
      }
    })
    val planner = clusters(0).getPlanner
    val defaultRelOptSchema = relOptSchemas(0).asInstanceOf[CalciteCatalogReader]

    // create Flink type factory
    val typeSystem = config.getTypeSystem
    val typeFactory = null// new FlinkTypeFactory(typeSystem)

    // create context instances with Flink type factory
    val cluster = RelOptCluster.create(planner, new RexBuilder(typeFactory))
    val calciteSchema = CalciteSchema.from(config.getDefaultSchema)
    val relOptSchema = new CalciteCatalogReader(
      calciteSchema,
      defaultRelOptSchema.getSchemaPaths.get(0),
      typeFactory,
      connectionConfig(config.getParserConfig))

    new SimpleRelBuilder(config.getContext, cluster, relOptSchema)
  }

  def connectionConfig(parserConfig : SqlParser.Config): CalciteConnectionConfig = {
    val prop = new Properties()
    prop.setProperty(CalciteConnectionProperty.CASE_SENSITIVE.camelName,
      String.valueOf(parserConfig.caseSensitive))
    new CalciteConnectionConfigImpl(prop)
  }

}
