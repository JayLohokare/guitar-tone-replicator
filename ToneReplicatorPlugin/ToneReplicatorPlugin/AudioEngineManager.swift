// AudioEngineManager.swift
// Manages the AVAudioEngine + AUv3 connection for standalone mode
// MIT License

import Foundation
import AVFoundation
import Combine

class AudioEngineManager: ObservableObject {
    @Published var isRunning = false
    @Published var inputGain: Float = 0.0      // dB
    @Published var outputGain: Float = 0.0     // dB
    @Published var dryWetMix: Float = 100.0    // percent
    @Published var bypass: Bool = false
    @Published var inputLevel: Float = 0.0     // 0-1
    @Published var outputLevel: Float = 0.0    // 0-1
    @Published var errorMessage: String?

    private var engine: AVAudioEngine?
    private var audioUnit: AUAudioUnit?
    private var tapNode: AVAudioNode?

    // Level metering timer
    private var meterTimer: Timer?

    func start() {
        guard !isRunning else { return }
        errorMessage = nil

        do {
            let engine = AVAudioEngine()
            self.engine = engine

            // Get the input node (microphone/audio interface)
            let inputNode = engine.inputNode
            let outputNode = engine.outputNode

            // Use the hardware format for the connection
            let hardwareFormat = inputNode.outputFormat(forBus: 0)

            // Install a tap on the input node to process audio
            let monoFormat = AVAudioFormat(standardFormatWithSampleRate: hardwareFormat.sampleRate,
                                            channels: 1)!

            inputNode.installTap(onBus: 0, bufferSize: 1024, format: monoFormat) { [weak self] buffer, time in
                self?.processAudioBuffer(buffer)
            }
            self.tapNode = inputNode

            // Connect input to output through the main mixer
            engine.connect(inputNode, to: engine.mainMixerNode, format: monoFormat)
            engine.connect(engine.mainMixerNode, to: outputNode, format: hardwareFormat)

            try engine.start()
            isRunning = true

            // Start level metering
            startMetering()

        } catch {
            errorMessage = "Failed to start audio engine: \(error.localizedDescription)"
            print("Audio engine error: \(error)")
        }
    }

    func stop() {
        guard isRunning else { return }

        // Remove tap
        if let tapNode = tapNode {
            tapNode.removeTap(onBus: 0)
            self.tapNode = nil
        }

        engine?.stop()
        engine = nil
        isRunning = false

        // Reset levels
        inputLevel = 0
        outputLevel = 0

        stopMetering()
    }

    func loadModel(url: URL) {
        // Model loading will be handled by the AUAudioUnit subclass
        // For standalone mode, we load CoreML models directly
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all

            // Compile the model if needed
            var modelURL = url
            if url.pathExtension == "mlpackage" {
                let compiledURL = try MLModel.compileModel(at: url)
                modelURL = compiledURL
            }

            let model = try MLModel(contentsOf: modelURL, configuration: config)
            print("Successfully loaded model from \(url.lastPathComponent)")

            // Store for inference
            self.mlModel = model

        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            print("Model load error: \(error)")
        }
    }

    // MARK: - Private

    private var mlModel: MLModel?
    private let processingQueue = DispatchQueue(label: "com.tonereplicator.processing", qos: .userInteractive)

    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameCount = Int(buffer.frameLength)

        // Update input level meter
        var inputRMS: Float = 0
        for i in 0..<frameCount {
            inputRMS += channelData[i] * channelData[i]
        }
        inputRMS = sqrt(inputRMS / Float(frameCount))

        DispatchQueue.main.async {
            self.inputLevel = min(inputRMS * 3.0, 1.0) // Scale for visibility
        }

        // Apply input gain
        let inputGainLinear = powf(10.0, inputGain * 0.05f)
        let outputGainLinear = powf(10.0, outputGain * 0.05f)
        let wetMix = dryWetMix / 100.0
        let dryMix = 1.0 - wetMix

        // Simple passthrough processing with gain
        for i in 0..<frameCount {
            let input = channelData[i] * inputGainLinear

            // For now, simple passthrough with gain until CoreML model is loaded
            var output: Float
            if mlModel != nil && !bypass {
                // CoreML processing would happen here
                // For now, apply gain and mix
                output = input * outputGainLinear
            } else {
                output = input * outputGainLinear
            }

            // Dry/wet mix
            let drySignal = input * outputGainLinear
            output = drySignal * dryMix + output * wetMix

            // Soft clip
            if abs(output) > 0.5 {
                output = tanhf(output)
            }

            channelData[i] = output
        }

        // Update output level meter
        var outputRMS: Float = 0
        for i in 0..<frameCount {
            outputRMS += channelData[i] * channelData[i]
        }
        outputRMS = sqrt(outputRMS / Float(frameCount))

        DispatchQueue.main.async {
            self.outputLevel = min(outputRMS * 3.0, 1.0)
        }
    }

    private func startMetering() {
        meterTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            // Level metering is handled in processAudioBuffer
        }
    }

    private func stopMetering() {
        meterTimer?.invalidate()
        meterTimer = nil
    }
}