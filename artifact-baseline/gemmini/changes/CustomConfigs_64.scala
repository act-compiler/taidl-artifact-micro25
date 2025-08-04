package gemmini

import org.chipsalliance.cde.config.{Config, Parameters}
import chisel3._
import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.subsystem.SystemBusKey
import freechips.rocketchip.tile.BuildRoCC


object GemminiCustomConfigs {
  // Default configurations
  val defaultConfig = GemminiConfigs.defaultConfig
  val defaultFpConfig = GemminiFPConfigs.defaultFPConfig

  // Create your own configs here
  val baselineInferenceConfig = defaultConfig.copy(
    has_training_convs = false,
  )

  val highPerfInferenceConfig = defaultConfig.copy(
    meshRows = 32,
    meshColumns = 32,

    has_training_convs = false,

    sp_capacity = CapacityInKilobytes(512),
    acc_capacity = CapacityInKilobytes(128),
  )

  val trainingConfig = defaultFpConfig.copy(
    inputType = Float(expWidth = 8, sigWidth = 24),
    accType = Float(expWidth = 8, sigWidth = 24),

    meshRows = 8,
    meshColumns = 8,

    has_training_convs = true,
    has_max_pool =  false,

    sp_capacity = CapacityInKilobytes(512),
    acc_capacity = CapacityInKilobytes(128),
  )

  val ibertInferenceConfig = defaultConfig.copy(
    has_training_convs = false,
    has_max_pool =  false,
    has_normalizations = true,

    acc_capacity = CapacityInKilobytes(128),
  )

  val SpikeConfig = defaultConfig.copy(
    meshRows = 64,
    meshColumns = 64,

    has_training_convs = false,

    sp_capacity = CapacityInKilobytes(1024),
    acc_capacity = CapacityInKilobytes(4096),
  )

  // Specify which of your custom configs you want to build here
  val customConfig = SpikeConfig
}


class GemminiCustomConfig[T <: Data : Arithmetic, U <: Data, V <: Data](
  gemminiConfig: GemminiArrayConfig[T,U,V] = GemminiCustomConfigs.customConfig
) extends Config((site, here, up) => {
  case BuildRoCC => up(BuildRoCC) ++ Seq(
    (p: Parameters) => {
      implicit val q = p
      val gemmini = LazyModule(new Gemmini(gemminiConfig))
      gemmini
    }
  )
})
