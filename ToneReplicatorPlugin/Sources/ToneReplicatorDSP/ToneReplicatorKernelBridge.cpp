// ToneReplicatorKernelBridge.cpp
// C bridge functions for Swift interop with the C++ DSP kernel
// MIT License

#include "ToneReplicatorKernel.hpp"

extern "C" {

void* toneReplicatorKernelCreate() {
    return new ToneReplicatorKernel();
}

void toneReplicatorKernelDestroy(void* kernel) {
    delete static_cast<ToneReplicatorKernel*>(kernel);
}

void toneReplicatorKernelInit(void* kernel, double sampleRate, uint32_t channelCount) {
    static_cast<ToneReplicatorKernel*>(kernel)->init(sampleRate, channelCount);
}

void toneReplicatorKernelReset(void* kernel) {
    static_cast<ToneReplicatorKernel*>(kernel)->reset();
}

void toneReplicatorKernelProcess(void* kernel, const float* inputBuffer,
                                  float* outputBuffer, uint32_t frameCount,
                                  uint32_t channelCount) {
    static_cast<ToneReplicatorKernel*>(kernel)->process(inputBuffer, outputBuffer,
                                                         frameCount, channelCount);
}

void toneReplicatorKernelSetInputGain(void* kernel, float gainDB) {
    static_cast<ToneReplicatorKernel*>(kernel)->setInputGain(gainDB);
}

void toneReplicatorKernelSetOutputGain(void* kernel, float gainDB) {
    static_cast<ToneReplicatorKernel*>(kernel)->setOutputGain(gainDB);
}

void toneReplicatorKernelSetDryWetMix(void* kernel, float mix) {
    static_cast<ToneReplicatorKernel*>(kernel)->setDryWetMix(mix);
}

void toneReplicatorKernelSetBypass(void* kernel, bool bypass) {
    static_cast<ToneReplicatorKernel*>(kernel)->setBypass(bypass);
}

void toneReplicatorKernelSetModelLoaded(void* kernel, bool loaded) {
    static_cast<ToneReplicatorKernel*>(kernel)->setModelLoaded(loaded);
}

bool toneReplicatorKernelIsModelLoaded(void* kernel) {
    return static_cast<ToneReplicatorKernel*>(kernel)->isModelLoaded();
}

} // extern "C"