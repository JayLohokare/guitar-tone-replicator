// ToneReplicatorKernel.cpp
// Real-time DSP kernel for Tone Replicator AUv3 plugin
// MIT License - Original code, no GPL dependencies

#include "ToneReplicatorKernel.hpp"
#include <algorithm>
#include <cstdlib>

ToneReplicatorKernel::ToneReplicatorKernel()
    : mSampleRate(44100.0)
    , mChannelCount(1)
    , mInputBuffer(MAX_BUFFER_SIZE, 0.0f)
    , mOutputBuffer(MAX_BUFFER_SIZE, 0.0f)
{
}

ToneReplicatorKernel::~ToneReplicatorKernel() {
}

void ToneReplicatorKernel::init(double sampleRate, uint32_t channelCount) {
    mSampleRate = sampleRate;
    mChannelCount = channelCount;
    reset();
}

void ToneReplicatorKernel::reset() {
    mInputBufferWritePos = 0;
    mInputBufferFill = 0;
    mOutputBufferReadPos = 0;
    mOutputBufferFill = 0;
    std::fill(mInputBuffer.begin(), mInputBuffer.end(), 0.0f);
    std::fill(mOutputBuffer.begin(), mOutputBuffer.end(), 0.0f);
}

float ToneReplicatorKernel::dbToLinear(float db) {
    // 0 dB = unity, -inf dB = silence
    if (db <= -120.0f) return 0.0f;
    return std::pow(10.0f, db * 0.05f);
}

float ToneReplicatorKernel::softClip(float sample) {
    // tanh soft clipper with linear region
    if (std::abs(sample) < 0.5f) return sample;
    return std::tanh(sample);
}

void ToneReplicatorKernel::crossfade(const float* dry, const float* wet,
                                      float* out, uint32_t length, float mix) {
    // Simple linear crossfade
    float wetGain = mix;
    float dryGain = 1.0f - mix;
    for (uint32_t i = 0; i < length; ++i) {
        out[i] = dryGain * dry[i] + wetGain * wet[i];
    }
}

void ToneReplicatorKernel::process(const float* inputBuffer, float* outputBuffer,
                                    uint32_t frameCount, uint32_t channelCount) {
    // Get current parameter values (atomic read)
    float inputGainLinear = dbToLinear(mInputGainDB.load(std::memory_order_relaxed));
    float outputGainLinear = dbToLinear(mOutputGainDB.load(std::memory_order_relaxed));
    float dryWetMix = mDryWetMix.load(std::memory_order_relaxed);
    bool bypass = mBypass.load(std::memory_order_relaxed);
    bool modelLoaded = mModelLoaded.load(std::memory_order_relaxed);

    // Simple passthrough when bypassed or no model loaded
    if (bypass || !modelLoaded) {
        for (uint32_t frame = 0; frame < frameCount; ++frame) {
            for (uint32_t ch = 0; ch < channelCount; ++ch) {
                uint32_t idx = frame * channelCount + ch;
                outputBuffer[idx] = inputBuffer[idx] * inputGainLinear * outputGainLinear;
            }
        }
        return;
    }

    // Process each sample frame
    // For now, we process mono (channel 0) through the model
    // and apply the same processing to all channels
    for (uint32_t frame = 0; frame < frameCount; ++frame) {
        // Get mono input sample
        float inputSample = 0.0f;
        for (uint32_t ch = 0; ch < channelCount; ++ch) {
            inputSample += inputBuffer[frame * channelCount + ch];
        }
        inputSample /= (float)channelCount;

        // Apply input gain
        float gainedInput = inputSample * inputGainLinear;

        // Store in input buffer for chunk processing
        mInputBuffer[mInputBufferWritePos] = gainedInput;
        mInputBufferWritePos = (mInputBufferWritePos + 1) % MAX_BUFFER_SIZE;
        mInputBufferFill = std::min(mInputBufferFill + 1, MAX_BUFFER_SIZE);

        // Try to get a processed output sample
        float wetSample = gainedInput; // Default: pass through
        if (mOutputBufferFill > 0) {
            wetSample = mOutputBuffer[mOutputBufferReadPos];
            mOutputBufferReadPos = (mOutputBufferReadPos + 1) % MAX_BUFFER_SIZE;
            mOutputBufferFill--;
        }

        // Crossfade dry/wet
        float drySample = gainedInput;
        float outputSample = drySample * (1.0f - dryWetMix) + wetSample * dryWetMix;

        // Apply output gain and soft clip
        outputSample = softClip(outputSample * outputGainLinear);

        // Write to all output channels
        for (uint32_t ch = 0; ch < channelCount; ++ch) {
            outputBuffer[frame * channelCount + ch] = outputSample;
        }
    }
}

void ToneReplicatorKernel::setInputGain(float gainDB) {
    mInputGainDB.store(gainDB, std::memory_order_relaxed);
}

void ToneReplicatorKernel::setOutputGain(float gainDB) {
    mOutputGainDB.store(gainDB, std::memory_order_relaxed);
}

void ToneReplicatorKernel::setDryWetMix(float mix) {
    mDryWetMix.store(std::clamp(mix, 0.0f, 1.0f), std::memory_order_relaxed);
}

void ToneReplicatorKernel::setBypass(bool bypass) {
    mBypass.store(bypass, std::memory_order_relaxed);
}

void ToneReplicatorKernel::setModelLoaded(bool loaded) {
    mModelLoaded.store(loaded, std::memory_order_relaxed);
}

bool ToneReplicatorKernel::isModelLoaded() const {
    return mModelLoaded.load(std::memory_order_relaxed);
}

bool ToneReplicatorKernel::processChunkWithCoreML(const float* input, float* output, uint32_t length) {
    // This method is called from Swift after CoreML inference
    // It's a bridge for the kernel to receive processed audio
    if (!mModelLoaded.load(std::memory_order_relaxed)) {
        return false;
    }

    // Write processed audio to the output ring buffer
    for (uint32_t i = 0; i < length; ++i) {
        uint32_t writePos = (mOutputBufferReadPos + mOutputBufferFill) % MAX_BUFFER_SIZE;
        mOutputBuffer[writePos] = input[i];
        if (mOutputBufferFill < MAX_BUFFER_SIZE) {
            mOutputBufferFill++;
        }
    }

    return true;
}