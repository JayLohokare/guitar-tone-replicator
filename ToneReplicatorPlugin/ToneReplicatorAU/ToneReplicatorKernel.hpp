// ToneReplicatorKernel.hpp
// Real-time DSP kernel for Tone Replicator AUv3 plugin
// MIT License - Original code, no GPL dependencies

#pragma once

#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <atomic>

class ToneReplicatorKernel {
public:
    ToneReplicatorKernel();
    ~ToneReplicatorKernel();

    // Initialize with sample rate and channel count
    void init(double sampleRate, uint32_t channelCount);

    // Reset internal state (called when plugin bypasses or format changes)
    void reset();

    // Process audio buffers (real-time safe, no allocations)
    // inputBuffer / outputBuffer: interleaved or mono float samples
    // frameCount: number of sample frames
    // channelCount: number of channels (should match init)
    void process(const float* inputBuffer, float* outputBuffer,
                 uint32_t frameCount, uint32_t channelCount);

    // Parameter setters (thread-safe via atomics)
    void setInputGain(float gainDB);
    void setOutputGain(float gainDB);
    void setDryWetMix(float mix);  // 0.0 = dry, 1.0 = wet
    void setBypass(bool bypass);

    // Model management
    void setModelLoaded(bool loaded);
    bool isModelLoaded() const;

    // CoreML inference bridge - called from Swift
    // Provides a chunk of mono samples for inference
    // Returns true if inference was performed, false if bypassed
    bool processChunkWithCoreML(const float* input, float* output, uint32_t length);

    // Receptive field size for the model
    static constexpr uint32_t RECEPTIVE_FIELD = 127;
    static constexpr uint32_t PROCESS_CHUNK_SIZE = 4096;

private:
    // Convert dB to linear gain
    static float dbToLinear(float db);

    // Crossfade between dry and wet signals
    void crossfade(const float* dry, const float* wet, float* out,
                   uint32_t length, float mix);

    // Apply gain with soft clipping to prevent harsh digital clipping
    float softClip(float sample);

    // State
    double mSampleRate;
    uint32_t mChannelCount;

    // Parameters (atomic for thread safety between UI and audio thread)
    std::atomic<float> mInputGainDB{0.0f};
    std::atomic<float> mOutputGainDB{0.0f};
    std::atomic<float> mDryWetMix{1.0f};
    std::atomic<bool> mBypass{false};
    std::atomic<bool> mModelLoaded{false};

    // Circular buffer for accumulating enough samples for model inference
    std::vector<float> mInputBuffer;
    uint32_t mInputBufferWritePos{0};
    uint32_t mInputBufferFill{0};

    // Output buffer for processed samples
    std::vector<float> mOutputBuffer;
    uint32_t mOutputBufferReadPos{0};
    uint32_t mOutputBufferFill{0};

    // Latency compensation: we buffer output to align dry/wet
    static constexpr uint32_t MAX_BUFFER_SIZE = 65536;
};