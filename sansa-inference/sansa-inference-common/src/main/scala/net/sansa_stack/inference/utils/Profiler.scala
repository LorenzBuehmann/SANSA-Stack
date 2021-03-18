package net.sansa_stack.inference.utils

trait Profiler extends Logging {

  def profile[R](block: => R): R = {
    val t0 = System.currentTimeMillis()
    val result = block
    // call-by-name
    val t1 = System.currentTimeMillis()
    info("Elapsed time: " + (t1 - t0) + "ms")
    result
  }
}
