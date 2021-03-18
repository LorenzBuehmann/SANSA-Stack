package net.sansa_stack.examples.flink.owl

import de.javakaffee.kryoserializers.UnmodifiableCollectionsSerializer
import net.sansa_stack.owl.flink.owl._
import org.apache.flink.api.scala.ExecutionEnvironment


object OWLReaderDataSet {

  def main(args: Array[String]) {
    parser.parse(args, Config()) match {
      case Some(config) =>
        run(config.in, config.syntax)
      case None =>
        println(parser.usage)
    }
  }

  def run(input: String, syntax: String): Unit = {

    println(".============================================.")
    println("| Dataset OWL reader example (" + syntax + " syntax)|")
    println(".============================================.")

    val env = ExecutionEnvironment.getExecutionEnvironment
    // scalastyle:off classforname
    env.getConfig.addDefaultKryoSerializer(
      Class.forName("java.util.Collections$UnmodifiableCollection"),
      classOf[UnmodifiableCollectionsSerializer])
    // scalastyle:on classforname

    val dataSet = syntax match {
      case "fun" => env.owl(Syntax.FUNCTIONAL)(input)
      case "manch" => env.owl(Syntax.MANCHESTER)(input)
      case "owl_xml" =>
        throw new RuntimeException("'" + syntax + "' - Not supported, yet.")
      case _ =>
        throw new RuntimeException("Invalid syntax type: '" + syntax + "'")
    }

    dataSet.first(10).print()

  }

  case class Config(
    in: String = "",
    syntax: String = "")

  // the CLI parser
  val parser = new scopt.OptionParser[Config]("Dataset OWL reader example") {

    head("Dataset OWL reader example")

    opt[String]('i', "input").required().valueName("<path>").
      action((x, c) => c.copy(in = x)).
      text("path to file that contains the data")

    opt[String]('s', "syntax").required().valueName("{fun | manch | owl_xml}").
      action((x, c) => c.copy(syntax = x)).
      text("the syntax format")

    help("help").text("prints this usage text")
  }
}
