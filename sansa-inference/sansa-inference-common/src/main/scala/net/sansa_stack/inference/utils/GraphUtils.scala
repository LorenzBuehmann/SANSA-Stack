package net.sansa_stack.inference.utils

import java.io.{ByteArrayOutputStream, File, FileOutputStream, FileWriter}
import java.util

import org.apache.jena.graph.Node
import org.apache.jena.reasoner.TriplePattern
import org.apache.jena.reasoner.rulesys.Rule
import org.apache.jena.shared.PrefixMapping
import org.apache.jena.sparql.util.FmtUtils
import org.jgrapht.Graph
import org.jgrapht.alg.isomorphism.VF2GraphIsomorphismInspector
import org.jgrapht.graph.{DefaultDirectedGraph, DirectedPseudograph}
import org.jgrapht.io.GraphMLExporter.AttributeCategory
import org.jgrapht.io._
import scalax.collection.edge.LDiEdge

import net.sansa_stack.inference.utils.graph.{EdgeEquivalenceComparator, LabeledEdge, NodeEquivalenceComparator}

/**
  * @author Lorenz Buehmann
  *         created on 1/23/16
  */
object GraphUtils {

  def areIsomorphic(graph1: scalax.collection.mutable.Graph[Node, LDiEdge], graph2: scalax.collection.mutable.Graph[Node, LDiEdge]): Boolean = {
    val g1 = asJGraphtRuleGraph(graph1)
    val g2 = asJGraphtRuleGraph(graph2)

    val c1 = new NodeEquivalenceComparator()
    val c2 = new EdgeEquivalenceComparator()

    val isoDetector = new VF2GraphIsomorphismInspector[Node, LabeledEdge[Node, String]](
      g1, g2, c1, c2)

    isoDetector.isomorphismExists()
  }

  /**
    * Converts a 'Graph for Scala' graph to a JGraphT graph.
    *
    * @param graph the 'Graph for Scala' graph
    * @return the JGraphT graph
    */
  def asJGraphtRuleGraph(graph: scalax.collection.mutable.Graph[Node, LDiEdge]): Graph[Node, LabeledEdge[Node, String]] = {
    val g: Graph[Node, LabeledEdge[Node, String]] = new DefaultDirectedGraph[Node, LabeledEdge[Node, String]](classOf[LabeledEdge[Node, String]])

    val edges = graph.edges.toList

    edges.foreach { e =>
      val nodes = e.nodes.toList
      val source = nodes(0)
      val target = nodes(1)
      g.addVertex(source)
      g.addVertex(target)

      // check if there is an edge e1 such that l(e)=source(e1) || l(e)=target(e2)
      // and if so set l(e)=l(e1)+"out"("in")
      var label = e.label.toString
      edges.foreach { e1 =>
        val nodes = e1.nodes.toList
        val source = nodes(0)
        val target = nodes(1)

        if(source.value.toString().equals(label)) {
          label = e1.label.toString + "_in"
        } else if (target.value.toString().equals(label)) {
          label = e1.label.toString + "_out"
        }
      }
      g.addEdge(source, target, new LabeledEdge[Node, String](source, target, label))
    }
    g
  }

  /**
    * Converts a 'Graph for Scala' graph to a JGraphT graph.
    *
    * @param graph the 'Graph for Scala' graph
    * @return the JGraphT graph
    */
  def asJGraphtRuleSetGraph(graph: scalax.collection.mutable.Graph[Rule, LDiEdge],
                            showInFlowDirection: Boolean = false): Graph[Rule, LabeledEdge[Rule, TriplePattern]] = {
    val g = new DefaultDirectedGraph[Rule, LabeledEdge[Rule, TriplePattern]](classOf[LabeledEdge[Rule, TriplePattern]])

    val edges = graph.edges.toList

    edges.foreach { e =>
      val s = e.source.value
      val t = e.target.value
      g.addVertex(s)
      g.addVertex(t)

      val label = e.label.asInstanceOf[TriplePattern]

      if (showInFlowDirection) {
        g.addEdge(t, s, LabeledEdge[Rule, TriplePattern](t, s, label))
      } else {
        g.addEdge(s, t, LabeledEdge[Rule, TriplePattern](s, t, label))
      }

    }

    g
  }



  implicit class ClassRuleDependencyGraphExporter(val graph: scalax.collection.mutable.Graph[Rule, LDiEdge]) {
    /**
      * Export the rule dependency graph to GraphML format.
      *
      * @param filename the target file
      */
    def export(filename: String, showInFlowDirection: Boolean = false,
               prefixMapping: PrefixMapping = PrefixMapping.Standard): Unit = {

      val g: Graph[Rule, LabeledEdge[Rule, TriplePattern]] = asJGraphtRuleSetGraph(graph, showInFlowDirection)

      // In order to be able to export edge and node labels and IDs,
      // we must implement providers for them
      val vertexIDProvider = new ComponentNameProvider[Rule]() {
        override def getName(v: Rule): String = v.getName
      }

      val vertexNameProvider = new ComponentNameProvider[Rule]() {
        override def getName(v: Rule): String = v.getName
      }

      val edgeIDProvider = new ComponentNameProvider[LabeledEdge[Rule, TriplePattern]]() {
        override def getName(edge: LabeledEdge[Rule, TriplePattern]): String = {
          g.getEdgeSource(edge) + " > " + g.getEdgeTarget(edge)
        }
      }

      val edgeLabelProvider = new ComponentNameProvider[LabeledEdge[Rule, TriplePattern]]() {
        override def getName(e: LabeledEdge[Rule, TriplePattern]): String = {
          val p = e.label.getPredicate
          // omit if predicate is a variable
          if(p.isVariable) {
            ""
          } else {
            FmtUtils.stringForNode(e.label.getPredicate, prefixMapping)
          }
        }
      }

      import org.jgrapht.io.DefaultAttribute
      val ruleDescriptionProvider = new ComponentAttributeProvider[Rule]() {
        override def getComponentAttributes(r: Rule): util.Map[String, Attribute] = {
          val map = new util.HashMap[String, Attribute]()
          map.put("rule", DefaultAttribute.createAttribute(r.toString))
          map
        }
      }

      //      val exporter = new GraphMLExporter[String,LabeledEdge](
//        vertexIDProvider, vertexNameProvider, edgeIDProvider,edgeLabelProvider)

      val exporter = new GraphMLExporter[Rule, LabeledEdge[Rule, TriplePattern]](
        new IntegerComponentNameProvider[Rule],
        vertexNameProvider,
        ruleDescriptionProvider,
        new IntegerComponentNameProvider[LabeledEdge[Rule, TriplePattern]],
        edgeLabelProvider,
        null)

      exporter.registerAttribute("rule", AttributeCategory.NODE, AttributeType.STRING)

      val fw = new FileWriter(filename)

      exporter.exportGraph(g, fw)
    }

  }

  implicit class ClassRuleTriplePatternGraphExporter(val graph: scalax.collection.mutable.Graph[Node, LDiEdge]) {


//    def export2(filename: String) = {
//
//      // Gephi
//      //Init a project - and therefore a workspace
//      val pc = Lookup.getDefault().lookup(classOf[ProjectController]);
//      pc.newProject();
//      val workspace = pc.getCurrentWorkspace();
//
//      //Get controllers and models
//      val importController = Lookup.getDefault().lookup(classOf[ImportController]);
//
//
//      //See if graph is well imported
//      val graphModel = Lookup.getDefault().lookup(classOf[GraphController]).getModel;
//      val g = graphModel.getDirectedGraph();
//
//      val edges = graph.edges.toList
//
//      edges.foreach { e =>
//        val nodes = e.nodes.toList
//        val source = graphModel.factory().newNode(nodes(0).toString())
//        val target = graphModel.factory().newNode(nodes(1).toString())
//        if(!g.contains(source))
//          g.addNode(source)
//        if(!g.contains(target))
//          g.addNode(target)
//        val edge = graphModel.factory().newEdge(source, target, 1.0f, true)
//        g.addEdge(edge)
//      }
//
//
//      //Run YifanHuLayout for 100 passes - The layout always takes the current visible view
//      val layout = new YifanHuLayout(null, new StepDisplacement(1f));
//      layout.setGraphModel(graphModel);
//      layout.resetPropertiesValues();
//      layout.setOptimalDistance(200f);
//
//      layout.initAlgo();
//      for (i <- 0 to 100 if layout.canAlgo()) {
//        layout.goAlgo();
//      }
//      layout.endAlgo();
//
//      val model = Lookup.getDefault().lookup(classOf[PreviewController]).getModel();
//      model.getProperties().putValue(PreviewProperty.SHOW_NODE_LABELS, true);
//      model.getProperties().putValue(PreviewProperty.EDGE_CURVED, false);
//      model.getProperties().putValue(PreviewProperty.EDGE_COLOR, new EdgeColor(java.awt.Color.GRAY));
//      model.getProperties().putValue(PreviewProperty.EDGE_THICKNESS, 0.1f);
//      model.getProperties().putValue(PreviewProperty.NODE_LABEL_FONT, model.getProperties().getFontValue(PreviewProperty.NODE_LABEL_FONT).deriveFont(8));
//      //      model.getProperties.putValue(Item.NODE_LABEL, "vertex_label")
//
//      for (item <- model.getItems(Item.NODE_LABEL)) {
//        println(item)
//      }
//
//
//      //Export full graph
//      val ec = Lookup.getDefault().lookup(classOf[ExportController]);
//      //      ec.exportFile(new File("io_gexf.gexf"));
//
//      //PDF Exporter config and export to Byte array
//      val pdfExporter = ec.getExporter("pdf").asInstanceOf[PDFExporter];
//      pdfExporter.setPageSize(PageSize.A0);
//      pdfExporter.setWorkspace(workspace);
//      val baos = new ByteArrayOutputStream();
//      ec.exportStream(baos, pdfExporter);
//      new FileOutputStream(filename + ".pdf").write(baos.toByteArray())
//    }

    /**
      * Export the rule dependency graph to GraphML format.
      *
      * @param filename the target file
      */
    def export(filename: String): Unit = {

      val g: Graph[Node, LabeledEdge[Node, Node]] = new DirectedPseudograph[Node, LabeledEdge[Node, Node]](classOf[LabeledEdge[Node, Node]])

      val edges = graph.edges.toList

      edges.foreach { e =>
        val s = e.source.value
        val t = e.target.value
        val label = e.label.asInstanceOf[Node]
        g.addVertex(s)
        g.addVertex(t)
        g.addEdge(s, t, LabeledEdge(s, t, label))
      }

      // In order to be able to export edge and node labels and IDs,
      // we must implement providers for them
      val vertexIDProvider = new ComponentNameProvider[Node]() {
        override def getName(v: Node): String = v.toString(PrefixMapping.Standard)
      }

      val vertexNameProvider = new ComponentNameProvider[Node]() {
        override def getName(v: Node): String = v.toString(PrefixMapping.Standard)
      }

      val edgeIDProvider = new ComponentNameProvider[LabeledEdge[Node, Node]]() {
        override def getName(edge: LabeledEdge[Node, Node]): String = {
          g.getEdgeSource(edge).toString(PrefixMapping.Standard) + " > " + edge.label + " > " + g.getEdgeTarget(edge).toString(PrefixMapping.Standard)
        }
      }

      val edgeLabelProvider = new ComponentNameProvider[LabeledEdge[Node, Node]]() {
        override def getName(e: LabeledEdge[Node, Node]): String = e.label.toString
      }

      //      val exporter = new GraphMLExporter[String,LabeledEdge](
      //        vertexIDProvider, vertexNameProvider, edgeIDProvider,edgeLabelProvider)

      val exporter = new GraphMLExporter[Node, LabeledEdge[Node, Node]](
        new IntegerComponentNameProvider[Node],
        vertexNameProvider,
        new IntegerComponentNameProvider[LabeledEdge[Node, Node]],
        edgeLabelProvider)

      val fw = new FileWriter(filename)

      exporter.exportGraph(g, fw)

//      val path = Paths.get(filename)
//      val charset = StandardCharsets.UTF_8
//
//      var content = new String(Files.readAllBytes(path), charset)
//      content = content.replaceAll("vertex_label", "node_label")
//      Files.write(path, content.getBytes(charset))
//
//      // Gephi
//      // Init a project - and therefore a workspace
//      val pc = Lookup.getDefault.lookup(classOf[ProjectController])
//      pc.newProject()
//      val workspace = pc.getCurrentWorkspace
//
//      // Get controllers and models
//      val importController = Lookup.getDefault.lookup(classOf[ImportController])
//
//      // Import file
//      val file = new File(filename)
//      var container = importController.importFile(file)
//      container.getLoader.setEdgeDefault(EdgeDirectionDefault.DIRECTED);   // Force DIRECTED
//      //        container.setAllowAutoNode(false);  //Don't create missing nodes
//
//      // Append imported data to GraphAPI
//      importController.process(container, new DefaultProcessor(), workspace)
//
//      // See if graph is well imported
//      val graphModel = Lookup.getDefault.lookup(classOf[GraphController]).getGraphModel
//      val diGraph = graphModel.getDirectedGraph()
//
//      for(node <- diGraph.getNodes.asScala) {
//        node.setLabel(node.getAttribute("node_label").toString)
//      }
//
//      for(edge <- diGraph.getEdges.asScala) {
//        edge.setLabel(edge.getAttribute("edge_label").toString)
//      }
//
//      // Run YifanHuLayout for 100 passes - The layout always takes the current visible view
//      val layout = new YifanHuLayout(null, new StepDisplacement(1f))
//      layout.setGraphModel(graphModel)
//      layout.resetPropertiesValues()
//      layout.setOptimalDistance(200f)
//
//      //      layout.initAlgo();
//      for (i <- 0 to 100 if layout.canAlgo()) {
//        layout.goAlgo();
//      }
//      layout.endAlgo();
//
//      val model = Lookup.getDefault.lookup(classOf[PreviewController]).getModel()
//      model.getProperties.putValue(PreviewProperty.SHOW_NODE_LABELS, true)
//      model.getProperties.putValue(PreviewProperty.EDGE_CURVED, false)
//      model.getProperties.putValue(PreviewProperty.EDGE_COLOR, new EdgeColor(java.awt.Color.GRAY))
//      model.getProperties.putValue(PreviewProperty.EDGE_THICKNESS, 0.1f)
//      model.getProperties.putValue(PreviewProperty.NODE_LABEL_FONT, model.getProperties.getFontValue(PreviewProperty.NODE_LABEL_FONT).deriveFont(8))
//
//      model.getProperties.putValue(Item.NODE_LABEL, "Vertex Label")
//      model.getProperties.putValue(PreviewProperty.SHOW_EDGE_LABELS, true)
//      model.getProperties.putValue(PreviewProperty.NODE_LABEL_SHOW_BOX, false)
//
//
//
//      // Export full graph
//      val ec = Lookup.getDefault.lookup(classOf[ExportController])
//      //      ec.exportFile(new File("io_gexf.gexf"));
//
//      // PDF Exporter config and export to Byte array
//      val pdfExporter = ec.getExporter("pdf").asInstanceOf[PDFExporter]
//      pdfExporter.setPageSize(PageSize.A0)
//      pdfExporter.setWorkspace(workspace)
//      val baos = new ByteArrayOutputStream()
//      ec.exportStream(baos, pdfExporter)
//      new FileOutputStream(filename + ".pdf").write(baos.toByteArray)
    }
  }
}
