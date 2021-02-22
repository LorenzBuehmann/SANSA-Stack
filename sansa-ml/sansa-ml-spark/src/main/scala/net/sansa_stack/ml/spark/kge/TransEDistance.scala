package net.sansa_stack.ml.spark.kge

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table
import scala.reflect.ClassTag

import com.intel.analytics.bigdl.nn.ErrorInfo

/**
 * @author Lorenz Buehmann
 */
class TransEDistance[T: ClassTag](
                                       val norm : Int = 2)
                                     (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

    override def updateOutput(input: Table): Tensor[T] = {
      var input1 = input[Tensor[T]](1).contiguous()
      var input2 = input[Tensor[T]](2).contiguous()
      var input3 = input[Tensor[T]](3).contiguous()

      output = input1 + input2 - input3

      output
    }

    private def mathsign(x: T): T = {
      if (ev.equals(x, ev.zero)) {
        2 * RNG.uniform(0, 2) - 3
      }

      if (ev.isGreater(x, ev.zero)) {
        ev.one
      } else {
        ev.negative(ev.one)
      }
    }

    override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
      require(input[Tensor[T]](1).dim() <= 2,
        "PairwiseDistance : " + ErrorInfo.constrainEachInputAsVectorOrBatch)

      if (!gradInput.contains(1)) {
        gradInput.update(1, Tensor[T]())
      }

      if (!gradInput.contains(2)) {
        gradInput.update(2, Tensor[T]())
      }

      gradInput[Tensor[T]](1).resizeAs(input[Tensor[T]](1))
      gradInput[Tensor[T]](2).resizeAs(input[Tensor[T]](2))

      gradInput[Tensor[T]](1)
        .copy(input[Tensor[T]](1))
        .add(ev.negative(ev.one), input[Tensor[T]](2))

      if (norm == 1) {
        gradInput[Tensor[T]](1).apply1(mathsign)
      } else {
        if (norm > 2) {
          gradInput[Tensor[T]](1)
            .cmul(gradInput[Tensor[T]](1)
              .clone()
              .abs()
              .pow(ev.minus(ev.fromType[Int](norm), ev.fromType[Int](2))))
        }
      }

      if (input[Tensor[T]](1).dim() > 1) {
        val outExpand = Tensor[T]()
        outExpand
          .resize(output.size(1), 1)
          .copy(output)
          .add(ev.fromType[Double](1.0e-6))
          .pow(ev.negative(ev.fromType[Int](norm - 1)))

        gradInput[Tensor[T]](1).cmul(
          outExpand.expand(
            Array(gradInput[Tensor[T]](1).size(1), gradInput[Tensor[T]](1).size(2))))
      } else {
        gradInput[Tensor[T]](1).mul(
          ev.pow(
            ev.plus(output.apply(Array(1)), ev.fromType[Double](1e-6)), ev.fromType[Int](1 - norm)))
      }

      if (input[Tensor[T]](1).dim() == 1) {
        gradInput[Tensor[T]](1).mul(gradOutput(Array(1)))
      } else {
        val grad = Tensor[T]()
        val ones = Tensor[T]()

        grad
          .resizeAs(input[Tensor[T]](1))
          .zero()

        ones
          .resize(input[Tensor[T]](1).size(2))
          .fill(ev.one)

        grad
          .addr(gradOutput, ones)
        gradInput[Tensor[T]](1).cmul(grad)
      }

      gradInput[Tensor[T]](2)
        .zero()
        .add(ev.negative(ev.one), gradInput[Tensor[T]](1))
      gradInput
    }

    override def toString: String = {
      s"nn.PairwiseDistance"
    }

    override def canEqual(other: Any): Boolean = other.isInstanceOf[TransEDistance[T]]


    override def equals(other: Any): Boolean = other match {
      case that: TransEDistance[T] =>
        super.equals(that) &&
          (that canEqual this) &&
          norm == that.norm
      case _ => false
    }

    override def hashCode(): Int = {
      val state = Seq(super.hashCode(), norm)
      state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
    }
  }

  object TransEDistance {
    def apply[@specialized(Float, Double) T: ClassTag](
                                                        norm : Int = 2)(implicit ev: TensorNumeric[T]) : TransEDistance[T] = {
      new TransEDistance[T](norm)
    }
  }

