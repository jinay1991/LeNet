///
/// @file
/// @copyright Copyright (c) 2020, MIT License
///
#ifndef INFERENCE_ENGINE_INFERENCE_ENGINE_INFERENCE_ENGINE_H_
#define INFERENCE_ENGINE_INFERENCE_ENGINE_INFERENCE_ENGINE_H_

#include "inference_engine/argument_parser/cli_options.h"
#include "inference_engine/inference_engine/i_inference_engine.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace inference_engine
{
/// @brief Inference Engine Strategy
class InferenceEngine
{
public:
  /// @brief Default Constructor
  InferenceEngine();

  /// @brief Destructor
  ~InferenceEngine() = default;

  /// @brief Initialise Inference Engine
  void Init();

  /// @brief Execute Inference with Inference Engine
  void Execute();

  /// @brief Release Inference Engine
  void Shutdown();

  /// @brief Select Inference Engine
  /// @param inference_engine [in] - Inference Engine type (TF, TFLite)
  void SelectInferenceEngine(const InferenceEngineType &inference_engine, const std::string &configuration);

  /// @brief Set input data (waveform)
  /// @param waveform [in] - Waveform to be split
  void SetInputWaveform(const Waveform &waveform, const std::int32_t nb_frames, const std::int32_t nb_channels);

  /// @brief Obtain Results for provided input
  /// @return vector of pair of (confidence, label idx)
  Waveforms GetResults() const;

  /// @brief Get type of inference engine. Used to determine which inference engine it is.
  ///
  /// @return Inference Engine type (i.e. TensorFlow, TensorFlowLite, etc.)
  InferenceEngineType GetType() const;

  /// @brief Provide configurations set for the model
  ///
  /// @return configuration
  std::string GetConfiguration() const;

private:
  /// @brief Command Line Options
  CLIOptions cli_options_;

  /// @brief Inference Engine
  std::unique_ptr<IInferenceEngine> inference_engine_;
};
} // namespace inference_engine
#endif /// INFERENCE_ENGINE_INFERENCE_ENGINE_INFERENCE_ENGINE_H_
