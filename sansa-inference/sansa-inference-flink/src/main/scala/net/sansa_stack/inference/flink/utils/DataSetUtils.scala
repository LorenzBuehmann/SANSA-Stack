package net.sansa_stack.inference.flink.utils

import java.lang.Iterable

import org.apache.flink.api.common.functions.CoGroupFunction
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala.DataSet
import org.apache.flink.util.Collector

import scala.reflect.ClassTag

/**
  * Some utility operations on Flink DataSets.
  *
  * @author Lorenz Buehmann
  */
object DataSetUtils {

  implicit class DataSetOps[T: ClassTag : TypeInformation](dataset: DataSet[T]) {

    /**
      * Splits a DataSet into two parts based on the given filter function. Note, that filtering is done twice on the same
      * data twice, thus, caching beforehand is recommended!
      *
      * @param f the boolean filter function
      * @return two DataSets
      */
    def partitionBy(f: T => Boolean): (DataSet[T], DataSet[T]) = {
      val passes = dataset.filter(f)
      val fails = dataset.filter((e: T) => !f(e)) // Flink doesn't have filterNot
      (passes, fails)
    }

    /**
      * Returns a DataSet with the elements from this that are not in `other`.
      *
      * @param other the DataSet containing the element to be subtracted
      * @return the DataSet
      */
    def subtract(other: DataSet[T]): DataSet[T] = {
      dataset.coGroup(other).where("*").equalTo("*")(
        new MinusCoGroupFunction[T](true))
        .name("subtract")
    }

    import scala.reflect._
    /**
      * Returns a DataSet with the elements from this that are not in `other`.
      * A key selector function for both datasets has to be given.
      *
      * @param other the DataSet containing the element to be subtracted
      * @return the DataSet
      */
    def subtract[K: ClassTag : TypeInformation](other: DataSet[T], keySelectorThis: (T) => K, keySelectorOther: (T) => K): DataSet[T] = {

      val typeInfo = TypeInformation.of(classTag[K].runtimeClass).asInstanceOf[TypeInformation[K]]
      dataset.coGroup(other)
        .where(keySelectorThis)
        .equalTo(keySelectorOther)(typeInfo)(
        new MinusCoGroupFunction[T](true))
        .name("subtract")
    }



  }

}

class MinusCoGroupFunction[T: ClassTag: TypeInformation](all: Boolean) extends CoGroupFunction[T, T, T] {
  override def coGroup(first: Iterable[T], second: Iterable[T], out: Collector[T]): Unit = {
    if (first == null || second == null) return
    val leftIter = first.iterator
    val rightIter = second.iterator

    if (all) {
      while (rightIter.hasNext && leftIter.hasNext) {
        leftIter.next()
        rightIter.next()
      }

      while (leftIter.hasNext) {
        out.collect(leftIter.next())
      }
    } else {
      if (!rightIter.hasNext && leftIter.hasNext) {
        out.collect(leftIter.next())
      }
    }
  }
}
