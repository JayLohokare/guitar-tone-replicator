// ContentView.swift
// Main UI for Tone Replicator standalone app
// MIT License

import SwiftUI
import AVFoundation

struct ContentView: View {
    @EnvironmentObject var modelStore: ModelStore
    @EnvironmentObject var audioEngine: AudioEngineManager

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Image(systemName: "guitars")
                    .font(.title2)
                    .foregroundColor(.orange)
                Text("Tone Replicator")
                    .font(.title2)
                    .fontWeight(.bold)
                Spacer()
                Text("v1.0")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(Color(nsColor: .windowBackgroundColor))

            Divider()

            // Main content
            HSplitView {
                // Left panel - Model browser
                ModelBrowserPanel()
                    .frame(minWidth: 200, idealWidth: 250)

                // Right panel - Controls
                VStack(spacing: 16) {
                    // Audio I/O controls
                    AudioControlsPanel()

                    // Level meters
                    LevelMetersPanel()

                    Spacer()
                }
                .padding()
                .frame(minWidth: 300, idealWidth: 400)
            }

            // Status bar
            HStack {
                Circle()
                    .fill(audioEngine.isRunning ? Color.green : Color.red)
                    .frame(width: 8, height: 8)
                Text(audioEngine.isRunning ? "Running" : "Stopped")
                    .font(.caption)
                Spacer()
                if let model = modelStore.activeModel {
                    Text("Model: \(model.name)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                if let error = audioEngine.errorMessage {
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.red)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color(nsColor: .controlBackgroundColor))
        }
    }
}

// MARK: - Model Browser Panel

struct ModelBrowserPanel: View {
    @EnvironmentObject var modelStore: ModelStore
    @EnvironmentObject var audioEngine: AudioEngineManager
    @State private var showingConverter = false
    @State private var showingFilePicker = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Models")
                .font(.headline)
                .padding(.horizontal)

            // Model list
            List(modelStore.models, selection: $modelStore.selectedModelID) { model in
                ModelRow(model: model, isActive: modelStore.activeModel?.id == model.id)
                    .tag(model.id)
            }
            .listStyle(.sidebar)

            // Buttons
            VStack(spacing: 8) {
                Button(action: { showingFilePicker = true }) {
                    Label("Import Model", systemImage: "square.and.arrow.down")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)

                Button(action: { showingConverter = true }) {
                    Label("Convert .pth to CoreML", systemImage: "arrow.triangle.2.circlepath")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)

                Button(action: loadSelectedModel) {
                    Label("Load Model", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(modelStore.selectedModelID == nil)
            }
            .padding()
        }
        .sheet(isPresented: $showingConverter) {
            CoreMLConverterView()
        }
        .fileImporter(isPresented: $showingFilePicker,
                      allowedContentTypes: [.item],
                      allowsMultipleSelection: false) { result in
            handleFileImport(result)
        }
    }

    private func loadSelectedModel() {
        guard let modelID = modelStore.selectedModelID,
              let model = modelStore.models.first(where: { $0.id == modelID }) else { return }
        modelStore.activeModel = model
        audioEngine.loadModel(url: model.url)
    }

    private func handleFileImport(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            modelStore.importModel(from: url)
        case .failure(let error):
            print("File import failed: \(error)")
        }
    }
}

struct ModelRow: View {
    let model: ToneModel
    let isActive: Bool

    var body: some View {
        HStack {
            Image(systemName: isActive ? "checkmark.circle.fill" : "circle")
                .foregroundColor(isActive ? .green : .secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(model.name)
                    .font(.body)
                Text(model.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }
}

// MARK: - Audio Controls Panel

struct AudioControlsPanel: View {
    @EnvironmentObject var audioEngine: AudioEngineManager

    var body: some View {
        VStack(spacing: 16) {
            Text("Controls")
                .font(.headline)

            // Input Gain
            ParameterSlider(label: "Input Gain",
                          value: $audioEngine.inputGain,
                          range: -40...40,
                          unit: "dB")

            // Output Gain
            ParameterSlider(label: "Output Gain",
                          value: $audioEngine.outputGain,
                          range: -40...40,
                          unit: "dB")

            // Dry/Wet Mix
            ParameterSlider(label: "Dry/Wet Mix",
                          value: $audioEngine.dryWetMix,
                          range: 0...100,
                          unit: "%")

            // Bypass
            Toggle(isOn: $audioEngine.bypass) {
                Label("Bypass", systemImage: audioEngine.bypass ? "waveform.path.badge.minus" : "waveform.path")
            }
            .toggleStyle(.switch)

            // Start/Stop
            HStack {
                Button(action: audioEngine.start) {
                    Label("Start", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(audioEngine.isRunning)

                Button(action: audioEngine.stop) {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(!audioEngine.isRunning)
            }
        }
    }
}

struct ParameterSlider: View {
    let label: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    let unit: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .font(.subheadline)
                Spacer()
                Text(String(format: "%.1f \(unit)", value))
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .monospacedDigit()
            }
            Slider(value: $value, in: range)
        }
    }
}

// MARK: - Level Meters

struct LevelMetersPanel: View {
    @EnvironmentObject var audioEngine: AudioEngineManager

    var body: some View {
        VStack(spacing: 8) {
            Text("Levels")
                .font(.headline)

            HStack(spacing: 24) {
                // Input level
                VStack(spacing: 4) {
                    Text("IN")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    LevelMeter(level: audioEngine.inputLevel)
                        .frame(width: 20, height: 120)
                }

                // Output level
                VStack(spacing: 4) {
                    Text("OUT")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    LevelMeter(level: audioEngine.outputLevel)
                        .frame(width: 20, height: 120)
                }
            }
        }
    }
}

struct LevelMeter: View {
    let level: Float

    private var barHeight: CGFloat {
        CGFloat(max(0, min(1, level)))
    }

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .bottom) {
                Rectangle()
                    .fill(Color(nsColor: .controlBackgroundColor))
                    .border(Color.secondary.opacity(0.3))

                Rectangle()
                    .fill(gradient)
                    .frame(height: geo.size.height * barHeight)
            }
        }
    }

    private var gradient: LinearGradient {
        LinearGradient(
            colors: [.green, .yellow, .red],
            startPoint: .bottom,
            endPoint: .top
        )
    }
}

// MARK: - CoreML Converter View

struct CoreMLConverterView: View {
    @EnvironmentObject var modelStore: ModelStore
    @Environment(\.dismiss) private var dismiss

    @State private var sourcePath = ""
    @State private var modelName = ""
    @State private var isConverting = false
    @State private var conversionProgress: Float = 0
    @State private var errorMessage: String?
    @State private var showFilePicker = false

    var body: some View {
        VStack(spacing: 16) {
            Text("Convert PyTorch Model to CoreML")
                .font(.headline)

            Form {
                TextField("Model Name", text: $modelName)

                HStack {
                    TextField("Source .pth file path", text: $sourcePath)
                    Button("Browse...") {
                        showFilePicker = true
                    }
                }

                if isConverting {
                    ProgressView("Converting...", value: conversionProgress, total: 1.0)
                }

                if let error = errorMessage {
                    Text(error)
                        .foregroundColor(.red)
                        .font(.caption)
                }
            }

            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Convert") {
                    convertModel()
                }
                .buttonStyle(.borderedProminent)
                .disabled(sourcePath.isEmpty || modelName.isEmpty || isConverting)
            }
        }
        .padding()
        .frame(width: 450, height: 300)
        .fileImporter(isPresented: $showFilePicker,
                      allowedContentTypes: [.item],
                      allowsMultipleSelection: false) { result in
            if case .success(let url) = result {
                sourcePath = url.path
                if modelName.isEmpty {
                    modelName = url.deletingPathExtension().lastPathComponent
                }
            }
        }
    }

    private func convertModel() {
        isConverting = true
        conversionProgress = 0
        errorMessage = nil

        // Run conversion in background
        DispatchQueue.global(qos: .userInitiated).async {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")

            let scriptPath = Bundle.main.bundlePath + "/../Resources/convert_model.py"
            process.arguments = [scriptPath, "--input", sourcePath, "--name", modelName]

            // For now, use the CoreMLConverter tool
            let modelsDir = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("ToneReplicator")
                .appendingPathComponent("models")

            let outputPath = modelsDir
                .appendingPathComponent(modelName)
                .appendingPathComponent("model.mlpackage")

            process.arguments = [
                Bundle.main.path(forResource: "convert_model", ofType: "py") ?? "",
                "--input", sourcePath,
                "--output", outputPath.path,
                "--name", modelName
            ]

            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe

            do {
                try process.run()
                process.waitUntilExit()

                DispatchQueue.main.async {
                    isConverting = false
                    if process.terminationStatus == 0 {
                        modelStore.refreshModels()
                        dismiss()
                    } else {
                        errorMessage = "Conversion failed (exit code \(process.terminationStatus))"
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    isConverting = false
                    errorMessage = "Failed to run conversion: \(error.localizedDescription)"
                }
            }
        }
    }
}

// MARK: - Preview

#Preview {
    ContentView()
        .environmentObject(ModelStore())
        .environmentObject(AudioEngineManager())
}