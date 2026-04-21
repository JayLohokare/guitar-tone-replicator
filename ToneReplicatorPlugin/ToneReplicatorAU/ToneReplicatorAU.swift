// ToneReplicatorAU.swift
// AUv3 Audio Unit implementation for Tone Replicator
// MIT License - Original code, no GPL dependencies

import AVFoundation
import CoreML

// MARK: - Parameter IDs
enum ToneReplicatorParam: UInt32 {
    case inputGain = 0
    case outputGain = 1
    case dryWetMix = 2
    case bypass = 3
}

// MARK: - Audio Unit
class ToneReplicatorAU: AUAudioUnit {
    // Parameters
    private var _inputGainParam: AUParameter!
    private var _outputGainParam: AUParameter!
    private var _dryWetParam: AUParameter!
    private var _bypassParam: AUParameter!

    private var _parameterTree: AUParameterTree!
    private var _inputBus: AUAudioUnitBus!
    private var _outputBus: AUAudioUnitBus!

    // Kernel bridge for C++ DSP
    private var _kernel = KernelBridge()

    // CoreML inference
    private var _mlModel: MLModel?
    private var _modelLoaded = false
    private var _processingQueue = DispatchQueue(label: "com.tonereplicator.inference",
                                                  qos: .userInteractive)

    // Audio format
    private var _format: AVAudioFormat!

    // Buffer for CoreML input
    private var _coreMLInputBuffer: [Float] = []
    private var _coreMLOutputBuffer: [Float] = []

    // Receptive field for the model
    private let RECEPTIVE_FIELD = 127
    private let INFERENCE_CHUNK_SIZE = 4096

    // MARK: - Initialization

    override init(componentDescription: AudioComponentDescription,
                  options: AudioComponentInstantiationOptions = []) throws {
        try super.init(componentDescription: componentDescription, options: options)

        // Create audio format (44.1kHz, mono, float32)
        _format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!

        // Setup buses
        _inputBus = try AUAudioUnitBus(format: _format)
        _outputBus = try AUAudioUnitBus(format: _format)

        let busArray = AUAudioUnitBusArray(audioUnit: self, busType: .input, buses: [_inputBus])
        inputBusArray = busArray

        let outputBusArray = AUAudioUnitBusArray(audioUnit: self, busType: .output, buses: [_outputBus])
        outputBusArray = outputBusArray

        // Setup parameters
        setupParameters()

        // Initialize kernel
        _kernel.initialize(sampleRate: 44100.0, channelCount: 1)
    }

    private func setupParameters() {
        // Input Gain: -40 to +40 dB, default 0
        _inputGainParam = AUParameter(
            identifier: "inputGain",
            name: "Input Gain",
            address: ToneReplicatorParam.inputGain.rawValue,
            min: -40.0,
            max: 40.0,
            unit: .decibels,
            unitName: nil,
            flags: [.readable, .writable],
            valueStrings: nil,
            dependentParameters: nil
        )

        // Output Gain: -40 to +40 dB, default 0
        _outputGainParam = AUParameter(
            identifier: "outputGain",
            name: "Output Gain",
            address: ToneReplicatorParam.outputGain.rawValue,
            min: -40.0,
            max: 40.0,
            unit: .decibels,
            unitName: nil,
            flags: [.readable, .writable],
            valueStrings: nil,
            dependentParameters: nil
        )

        // Dry/Wet Mix: 0% to 100%, default 100%
        _dryWetParam = AUParameter(
            identifier: "dryWetMix",
            name: "Dry/Wet Mix",
            address: ToneReplicatorParam.dryWetMix.rawValue,
            min: 0.0,
            max: 100.0,
            unit: .percent,
            unitName: nil,
            flags: [.readable, .writable],
            valueStrings: nil,
            dependentParameters: nil
        )

        // Bypass: 0 or 1
        _bypassParam = AUParameter(
            identifier: "bypass",
            name: "Bypass",
            address: ToneReplicatorParam.bypass.rawValue,
            min: 0.0,
            max: 1.0,
            unit: .boolean,
            unitName: nil,
            flags: [.readable, .writable],
            valueStrings: nil,
            dependentParameters: nil
        )

        _parameterTree = AUParameterTree(
            identifier: "ToneReplicatorParams",
            name: "Tone Replicator Parameters",
            children: [_inputGainParam, _outputGainParam, _dryWetParam, _bypassParam],
            parameterObserver: { [weak self] address, value in
                self?.parameterChanged(address: address, value: value)
            }
        )

        parameterTree = _parameterTree

        // Set defaults
        _inputGainParam.value = 0.0
        _outputGainParam.value = 0.0
        _dryWetParam.value = 100.0
        _bypassParam.value = 0.0
    }

    private func parameterChanged(address: AUParameterAddress, value: Float) {
        switch address {
        case ToneReplicatorParam.inputGain.rawValue:
            _kernel.setInputGain(value)
        case ToneReplicatorParam.outputGain.rawValue:
            _kernel.setOutputGain(value)
        case ToneReplicatorParam.dryWetMix.rawValue:
            _kernel.setDryWetMix(value / 100.0) // Convert percent to 0-1
        case ToneReplicatorParam.bypass.rawValue:
            _kernel.setBypass(value > 0.5)
        default:
            break
        }
    }

    // MARK: - Audio Unit Lifecycle

    override var internalRenderBlock: AUInternalRenderBlock {
        return { [weak self] actionFlags, timestamp, frameCount, outputBusNumber, outputAudioBufferList,
                  triggerBlock, framesToProvide in

            guard let self = self else {
                // Passthrough on dealloc
                let outputBuffer = UnsafeMutableAudioBufferListPointer(outputAudioBufferList)[0]
                return noErr
            }

            let outputBuffers = UnsafeMutableAudioBufferListPointer(outputAudioBufferList)

            // Get input from the upstream connection
            if let inputBuffer = self.inputBusArray?.buses[0].mutableAudioBufferList {
                let inputBuffers = UnsafeMutableAudioBufferListPointer(inputBuffer)

                // For each channel
                for channel in 0..<Int(outputBuffers.count) {
                    guard let inputData = inputBuffers[channel].mData?.assumingMemoryBound(to: Float.self),
                          let outputData = outputBuffers[channel].mData?.assumingMemoryBound(to: Float.self) else {
                        continue
                    }

                    let inputFrameCount = inputBuffers[channel].mDataByteSize / UInt32(MemoryLayout<Float>.size)
                    let channelCount = UInt32(outputBuffers.count)

                    // Process through kernel
                    self._kernel.process(
                        inputBuffer: inputData,
                        outputBuffer: outputData,
                        frameCount: min(frameCount, inputFrameCount),
                        channelCount: channelCount
                    )
                }
            } else {
                // No input connected - zero the output
                for channel in 0..<Int(outputBuffers.count) {
                    guard let outputData = outputBuffers[channel].mData?.assumingMemoryBound(to: Float.self) else {
                        continue
                    }
                    for frame in 0..<Int(frameCount) {
                        outputData[frame] = 0.0
                    }
                }
            }

            return noErr
        }
    }

    // MARK: - Format Support

    override func supportedChannelCounts(forBus bus: AUAudioUnitBus) -> IndexSet {
        return IndexSet([1]) // Mono only
    }

    override func supportedSampleRates(forBus bus: AUAudioUnitBus) -> IndexSet {
        return IndexSet([44100, 48000])
    }

    // MARK: - Model Management

    func loadModel(from url: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all // Use ANE when available

        let model = try MLModel(contentsOf: url, configuration: config)

        // Test the model with a small input to verify it works
        let inputShape = try model.modelDescription.inputDescriptionsByName["input"]?.multiArrayConstraint?.shape
            ?? [1, 1, 2048] as [NSNumber]

        // Allocate inference buffers
        let chunkSize = INFERENCE_CHUNK_SIZE
        _coreMLInputBuffer = [Float](repeating: 0.0, count: chunkSize + RECEPTIVE_FIELD)
        _coreMLOutputBuffer = [Float](repeating: 0.0, count: chunkSize)

        _mlModel = model
        _modelLoaded = true
        _kernel.setModelLoaded(true)
    }

    func loadModel(name: String) throws {
        let modelsDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("ToneReplicator")
            .appendingPathComponent("models")
            .appendingPathComponent(name)

        // Try .mlpackage first
        let mlpackageURL = modelsDir.appendingPathComponent("model.mlpackage")
        if FileManager.default.fileExists(atPath: mlpackageURL.path) {
            try loadModel(from: mlpackageURL)
            return
        }

        // Try .mlmodelc (compiled CoreML model)
        let mlmodelcURL = modelsDir.appendingPathComponent("model.mlmodelc")
        if FileManager.default.fileExists(atPath: mlmodelcURL.path) {
            try loadModel(from: mlmodelcURL)
            return
        }

        throw ToneReplicatorError.modelNotFound(name: name)
    }

    // MARK: - CoreML Inference

    func runInference(inputSamples: [Float]) -> [Float]? {
        guard let model = _mlModel else { return nil }

        do {
            let inputArray = try MLMultiArray(shape: [1, 1, NSNumber(value: inputSamples.count)],
                                                dataType: .float32)
            for i in 0..<inputSamples.count {
                inputArray[i] = NSNumber(value: inputSamples[i])
            }

            let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
                "input": inputArray
            ])

            let result = try model.prediction(from: inputProvider)

            guard let outputArray = result.featureValue(for: "output")?.multiArrayValue else {
                return nil
            }

            var output = [Float](repeating: 0.0, count: Int(outputArray.count))
            for i in 0..<output.count {
                output[i] = outputArray[i].floatValue
            }

            return output
        } catch {
            print("ToneReplicator: CoreML inference failed: \(error)")
            return nil
        }
    }
}

// MARK: - Errors

enum ToneReplicatorError: LocalizedError {
    case modelNotFound(name: String)
    case inferenceFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Model not found: \(name)"
        case .inferenceFailed(let message):
            return "Inference failed: \(message)"
        }
    }
}