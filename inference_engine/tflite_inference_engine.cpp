///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#include "inference_engine/inference_engine/tflite_inference_engine.h"

#include "inference_engine/logging/logging.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include <sys/time.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace inference_engine
{
namespace
{
/// @brief Convert to usec (microseconds)
constexpr double get_us(struct timeval t)
{
    return (t.tv_sec * 1000000 + t.tv_usec);
}

inline std::ostream &operator<<(std::ostream &os, const TfLiteIntArray *v)
{
    if (!v)
    {
        os << " (null)";
        return os;
    }
    for (int k = 0; k < v->size; k++)
    {
        os << " " << std::dec << std::setw(4) << v->data[k];
    }
    return os;
}

inline std::ostream &operator<<(std::ostream &os, const TfLiteType &type)
{
    switch (type)
    {
    case kTfLiteFloat32:
        os << "float32";
        break;
    case kTfLiteInt32:
        os << "int32";
        break;
    case kTfLiteUInt8:
        os << "uint8";
        break;
    case kTfLiteInt8:
        os << "int8";
        break;
    case kTfLiteInt64:
        os << "int64";
        break;
    case kTfLiteString:
        os << "string";
        break;
    case kTfLiteBool:
        os << "bool";
        break;
    case kTfLiteInt16:
        os << "int16";
        break;
    case kTfLiteComplex64:
        os << "complex64";
        break;
    case kTfLiteFloat16:
        os << "float16";
        break;
    case kTfLiteNoType:
        os << "no type";
        break;
    default:
        os << "(invalid)";
        break;
    }
    return os;
}

} // namespace

TFLiteInferenceEngine::TFLiteInferenceEngine() : TFLiteInferenceEngine{"inference_engine:5stems"} {}

TFLiteInferenceEngine::TFLiteInferenceEngine(const std::string &configuration) : configuration_{configuration} {}

void TFLiteInferenceEngine::Init()
{
    model_ = tflite::FlatBufferModel::BuildFromFile(GetModelPath().c_str());
    ASSERT_CHECK(model_) << "Failed to mmap model " << GetModelPath();
    model_->error_reporter();

    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
    ASSERT_CHECK(interpreter_) << "Failed to construct interpreter";

    INFERENCE_ENGINE_LOG(INFO) << "tensors size: " << interpreter_->tensors_size();
    INFERENCE_ENGINE_LOG(INFO) << "nodes size: " << interpreter_->nodes_size();
    INFERENCE_ENGINE_LOG(INFO) << "inputs: " << interpreter_->inputs().size();
    INFERENCE_ENGINE_LOG(INFO) << "input(0) name: " << interpreter_->GetInputName(0);

    int tensor_size = interpreter_->tensors_size();
    for (int i = 0; i < tensor_size; i++)
    {
        if (interpreter_->tensor(i)->name)
        {
            INFERENCE_ENGINE_LOG(INFO) << i << ": " << interpreter_->tensor(i)->name << ", " << interpreter_->tensor(i)->bytes
                                       << ", " << interpreter_->tensor(i)->type << ", " << interpreter_->tensor(i)->params.scale
                                       << ", " << interpreter_->tensor(i)->params.zero_point;
        }
    }

    const auto inputs = interpreter_->inputs();
    const auto outputs = interpreter_->outputs();
    INFERENCE_ENGINE_LOG(INFO) << "number of inputs: " << inputs.size();
    INFERENCE_ENGINE_LOG(INFO) << "number of outputs: " << outputs.size();

    ASSERT_CHECK_EQ(interpreter_->AllocateTensors(), TfLiteStatus::kTfLiteOk) << "Failed to allocate tensors!";

    PrintInterpreterState(interpreter_.get());
}

void TFLiteInferenceEngine::Execute()
{
    SetInputWaveform(Waveform{}, 0, 0);
    results_ = InvokeInference();
}

void TFLiteInferenceEngine::Shutdown() {}

Waveforms TFLiteInferenceEngine::InvokeInference() const
{
    auto error_code = interpreter_->Invoke();
    ASSERT_CHECK_EQ(error_code, TfLiteStatus::kTfLiteOk) << "Failed to invoke tflite!";

    return Waveforms{};
}

void TFLiteInferenceEngine::SetInputWaveform(const Waveform &waveform,
                                             const std::int32_t nb_frames,
                                             const std::int32_t nb_channels)
{
}

Waveforms TFLiteInferenceEngine::GetResults() const
{
    return results_;
}

std::string TFLiteInferenceEngine::GetModelPath() const
{
    return "data/model.t";
}

InferenceEngineType TFLiteInferenceEngine::GetType() const
{
    return InferenceEngineType::kTensorFlowLite;
};

std::string TFLiteInferenceEngine::GetConfiguration() const
{
    return configuration_;
}
} // namespace inference_engine
