package net.sansa_stack.rdf.flink.qualityassessment.metrics.licensing

import net.sansa_stack.rdf.common.qualityassessment.utils.NodeUtils._
import org.apache.flink.api.scala.DataSet
import org.apache.jena.graph.Triple

/**
 * @author Gezim Sejdiu
 */
object HumanReadableLicense {

  /**
   * Human -readable indication of a license
   * This metric checks whether a human-readable text, stating the of licensing model
   * attributed to the resource, has been provided as part of the dataset.
   * It looks for objects containing literal values and analyzes the text searching for key, licensing related terms.
   */
  def assessHumanReadableLicense(dataset: DataSet[Triple]): Double = {

    val hasValidLicense = dataset.filter { f =>
      f.getSubject.isURI() && hasLicenceIndications(f.getPredicate) &&
        f.getObject.isLiteral() && isLicenseStatement(f.getObject)
    }

    if (hasValidLicense.count() > 0) 1.0 else 0.0
  }

}
