// ToneReplicatorBridge.swift
// Bridge between Swift AUAudioUnit and C++ DSP kernel
// MIT License

import Foundation

// This bridge connects the Swift AUv3 world to our C++ DSP kernel.
// The C++ kernel handles real-time audio processing with no allocations.

@_silgen_name("toneReplicatorKernelCreate")
public func toneReplicatorKernelCreate() -> UnsafeMutableRawPointer

@_silgen_name("toneReplicatorKernelDestroy")
public func toneReplicatorKernelDestroy(_ kernel: UnsafeMutableRawPointer)

@_silgen_name("toneReplicatorKernelInit")
public func toneReplicatorKernelInit(_ kernel: UnsafeMutableRawPointer, sampleRate: Double, channelCount: UInt32)

@_silgen_name("toneReplicatorKernelReset")
public func toneReplicatorKernelReset(_ kernel: UnsafeMutableRawPointer)

@_silgen_name("toneReplicatorKernelProcess")
public func toneReplicatorKernelProcess(_ kernel: UnsafeMutableRawPointer,
                                         inputBuffer: UnsafePointer<Float>,
                                         outputBuffer: UnsafeMutablePointer<Float>,
                                         frameCount: UInt32,
                                         channelCount: UInt32)

@_silgen_name("toneReplicatorKernelSetInputGain")
public func toneReplicatorKernelSetInputGain(_ kernel: UnsafeMutableRawPointer, gainDB: Float)

@_silgen_name("toneReplicatorKernelSetOutputGain")
public func toneReplicatorKernelSetOutputGain(_ kernel: UnsafeMutableRawPointer, gainDB: Float)

@_silgen_name("toneReplicatorKernelSetDryWetMix")
public func toneReplicatorKernelSetDryWetMix(_ kernel: UnsafeMutableRawPointer, mix: Float)

@_silgen_name("toneReplicatorKernelSetBypass")
public func toneReplicatorKernelSetBypass(_ kernel: UnsafeMutableRawPointer, bypass: Bool)

@_silgen_name("toneReplicatorKernelSetModelLoaded")
public func toneReplicatorKernelSetModelLoaded(_ kernel: UnsafeMutableRawPointer, loaded: Bool)

@_silgen_name("toneReplicatorKernelIsModelLoaded")
public func toneReplicatorKernelIsModelLoaded(_ kernel: UnsafeMutableRawPointer) -> Bool

/// Swift wrapper around the C++ kernel for easier use from the AUAudioUnit subclass
class KernelBridge {
    private var kernel: UnsafeMutableRawPointer?

    init() {
        kernel = toneReplicatorKernelCreate()
    }

    deinit {
        if let k = kernel {
            toneReplicatorKernelDestroy(k)
        }
    }

    func initialize(sampleRate: Double, channelCount: UInt32) {
        guard let k = kernel else { return }
        toneReplicatorKernelInit(k, sampleRate: sampleRate, channelCount: channelCount)
    }

    func reset() {
        guard let k = kernel else { return }
        toneReplicatorKernelReset(k)
    }

    func process(inputBuffer: UnsafePointer<Float>, outputBuffer: UnsafeMutablePointer<Float>,
                 frameCount: UInt32, channelCount: UInt32) {
        guard let k = kernel else { return }
        toneReplicatorKernelProcess(k, inputBuffer: inputBuffer, outputBuffer: outputBuffer,
                                    frameCount: frameCount, channelCount: channelCount)
    }

    func setInputGain(_ gainDB: Float) {
        guard let k = kernel else { return }
        toneReplicatorKernelSetInputGain(k, gainDB: gainDB)
    }

    func setOutputGain(_ gainDB: Float) {
        guard let k = kernel else { return }
        toneReplicatorKernelSetOutputGain(k, gainDB: gainDB)
    }

    func setDryWetMix(_ mix: Float) {
        guard let k = kernel else { return }
        toneReplicatorKernelSetDryWetMix(k, mix: mix)
    }

    func setBypass(_ bypass: Bool) {
        guard let k = kernel else { return }
        toneReplicatorKernelSetBypass(k, bypass: bypass)
    }

    func setModelLoaded(_ loaded: Bool) {
        guard let k = kernel else { return }
        toneReplicatorKernelSetModelLoaded(k, loaded: loaded)
    }

    var isModelLoaded: Bool {
        guard let k = kernel else { return false }
        return toneReplicatorKernelIsModelLoaded(k)
    }
}