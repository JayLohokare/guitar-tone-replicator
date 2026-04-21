import SwiftUI
import AVFoundation
import Combine

// MARK: - Data Models

struct ToneModel: Codable, Identifiable {
    let id: UUID
    let name: String
    let createdAt: Date
    let modelPath: String
    let sampleRate: Int
    let valEsr: Double
    let sourceSong: String
    
    var displayName: String {
        "\(name) (ESR: \(String(format: "%.4f", valEsr)))"
    }
}

struct TrainingProgress: Codable {
    let step: String
    let epoch: Int?
    let trainEsr: Double?
    let valEsr: Double?
    let progress: Int?
}

struct ToneJob: Identifiable {
    let id = UUID()
    var status: JobStatus
    var progress: Double
    var message: String
    var model: ToneModel?
    
    enum JobStatus {
        case idle, separating, preparing, training, complete, error
    }
}

// MARK: - Backend API Client

class ToneReplicatorAPI: ObservableObject {
    static let shared = ToneReplicatorAPI()
    
    private let baseURL = "http://localhost:8767"
    private let session = URLSession.shared
    
    @Published var isConnected = false
    @Published var models: [ToneModel] = []
    @Published var currentJob = ToneJob(status: .idle, progress: 0, message: "Ready")
    
    private var progressTimer: Timer?
    
    func checkConnection() {
        guard let url = URL(string: "\(baseURL)/health") else { return }
        session.dataTask(with: url) { [weak self] data, response, error in
            DispatchQueue.main.async {
                self?.isConnected = (response as? HTTPURLResponse)?.statusCode == 200 ?? false
            }
        }.resume()
    }
    
    func startTraining(songPath: String, modelName: String) {
        guard let url = URL(string: "\(baseURL)/train") else { return }
        
        DispatchQueue.main.async {
            self.currentJob = ToneJob(status: .separating, progress: 0, message: "Separating guitar stem...")
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body: [String: String] = [
            "song_path": songPath,
            "model_name": modelName,
            "model_type": "wavenet",
            "model_size": "standard",
            "epochs": "100"
        ]
        request.httpBody = try? JSONEncoder().encode(body)
        
        session.dataTask(with: request) { [weak self] data, response, error in
            guard let data = data,
                  let result = try? JSONDecoder().decode([String: String].self, from: data),
                  let jobId = result["job_id"] else {
                DispatchQueue.main.async {
                    self?.currentJob = ToneJob(status: .error, progress: 0, message: error?.localizedDescription ?? "Failed to start training")
                }
                return
            }
            
            DispatchQueue.main.async {
                self?.pollProgress(jobId: jobId)
            }
        }.resume()
    }
    
    private func pollProgress(jobId: String) {
        progressTimer?.invalidate()
        progressTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] timer in
            guard let url = URL(string: "\(self?.baseURL ?? "")/progress/\(jobId)") else {
                timer.invalidate()
                return
            }
            
            self?.session.dataTask(with: url) { data, response, error in
                guard let data = data,
                      let progress = try? JSONDecoder().decode(TrainingProgress.self, from: data) else { return }
                
                DispatchQueue.main.async {
                    switch progress.step {
                    case "separating":
                        self?.currentJob.status = .separating
                        self?.currentJob.message = "Separating guitar stem..."
                        self?.currentJob.progress = Double(progress.progress ?? 0) / 100.0 * 0.2
                    case "preparing":
                        self?.currentJob.status = .preparing
                        self?.currentJob.message = "Preparing training data..."
                        self?.currentJob.progress = 0.2 + Double(progress.progress ?? 0) / 100.0 * 0.1
                    case "training":
                        self?.currentJob.status = .training
                        if let epoch = progress.epoch, let valEsr = progress.valEsr {
                            self?.currentJob.message = "Training epoch \(epoch): ESR = \(String(format: "%.4f", valEsr))"
                            self?.currentJob.progress = 0.3 + Double(epoch) / 100.0 * 0.7
                        }
                    case "complete":
                        self?.currentJob.status = .complete
                        self?.currentJob.message = "Training complete!"
                        self?.currentJob.progress = 1.0
                        timer.invalidate()
                        self?.loadModels()
                    case "error":
                        self?.currentJob.status = .error
                        self?.currentJob.message = progress.message
                        timer.invalidate()
                    default:
                        break
                    }
                }
            }.resume()
        }
    }
    
    func loadModels() {
        guard let url = URL(string: "\(baseURL)/models") else { return }
        session.dataTask(with: url) { [weak self] data, response, error in
            guard let data = data else { return }
            DispatchQueue.main.async {
                self?.models = (try? JSONDecoder().decode([ToneModel].self, from: data)) ?? []
            }
        }.resume()
    }
    
    func applyTone(modelId: UUID, diPath: String, outputPath: String) {
        guard let url = URL(string: "\(baseURL)/apply") else { return }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body: [String: String] = [
            "model_id": modelId.uuidString,
            "di_path": diPath,
            "output_path": outputPath,
        ]
        request.httpBody = try? JSONEncoder().encode(body)
        
        session.dataTask(with: request) { data, response, error in
            // Handle response
        }.resume()
    }
}

// MARK: - Main App

@main
struct ToneReplicatorApp: App {
    @StateObject private var api = ToneReplicatorAPI.shared
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(api)
                .onAppear {
                    api.checkConnection()
                    api.loadModels()
                }
        }
        .windowStyle(.titleBar)
        .windowResizability(.contentSize)
    }
}

// MARK: - Content View

struct ContentView: View {
    @EnvironmentObject var api: ToneReplicatorAPI
    @State private var selectedTab = 0
    @State private var songPath = ""
    @State private var modelName = ""
    @State private var diPath = ""
    @State private var outputPath = ""
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Image(systemName: "guitars")
                    .font(.title2)
                    .foregroundStyle(.orange)
                Text("Tone Replicator")
                    .font(.title2.bold())
                Spacer()
                Circle()
                    .fill(api.isConnected ? .green : .red)
                    .frame(width: 10, height: 10)
                Text(api.isConnected ? "Connected" : "Offline")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding()
            .background(.ultraThinMaterial)
            
            Divider()
            
            // Tab View
            TabView(selection: $selectedTab) {
                TrainView()
                    .tabItem {
                        Label("Train", systemImage: "waveform.badge.magnifyingglass")
                    }
                    .tag(0)
                
                ApplyView()
                    .tabItem {
                        Label("Apply", systemImage: "music.note")
                    }
                    .tag(1)
                
                ModelsView()
                    .tabItem {
                        Label("Models", systemImage: "externaldrive")
                    }
                    .tag(2)
            }
        }
        .frame(minWidth: 600, minHeight: 500)
    }
}

// MARK: - Train View

struct TrainView: View {
    @EnvironmentObject var api: ToneReplicatorAPI
    @State private var songPath = ""
    @State private var modelName = "MyTone"
    @State private var modelSize = "standard"
    @State private var epochs = 100
    @State private var isTraining = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Train a Tone Model")
                .font(.headline)
            
            Text("Point to a song, and we'll extract the guitar tone and train a model that replicates it.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            
            // Song file picker
            HStack {
                TextField("Song file path...", text: $songPath)
                    .textFieldStyle(.roundedBorder)
                
                Button("Browse") {
                    let panel = NSOpenPanel()
                    panel.allowedContentTypes = [.audio]
                    panel.allowsMultipleSelection = false
                    if panel.runModal() == .OK, let url = panel.url {
                        songPath = url.path
                    }
                }
            }
            
            // Model name
            TextField("Model name", text: $modelName)
                .textFieldStyle(.roundedBorder)
            
            // Settings
            HStack {
                Picker("Model Size", selection: $modelSize) {
                    Text("Nano (fastest)").tag("nano")
                    Text("Lite").tag("lite")
                    Text("Standard (recommended)").tag("standard")
                }
                .frame(width: 280)
                
                Stepper("Epochs: \(epochs)", value: $epochs, in: 10...500, step: 10)
            }
            
            // Train button
            Button(action: startTraining) {
                HStack {
                    Image(systemName: isTraining ? "arrow.trianglehead.2.clockwise" : "waveform.badge.magnifyingglass")
                    Text(isTraining ? "Training..." : "Train Tone Model")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(songPath.isEmpty || isTraining)
            .tint(.orange)
            
            // Progress
            if api.currentJob.status != .idle {
                VStack(alignment: .leading, spacing: 8) {
                    ProgressView(value: api.currentJob.progress)
                        .progressViewStyle(.linear)
                    Text(api.currentJob.message)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 8)
            }
            
            Spacer()
        }
        .padding()
    }
    
    private func startTraining() {
        isTraining = true
        api.startTraining(songPath: songPath, modelName: modelName)
    }
}

// MARK: - Apply View

struct ApplyView: View {
    @EnvironmentObject var api: ToneReplicatorAPI
    @State private var selectedModel: ToneModel?
    @State private var diPath = ""
    @State private var outputPath = ""
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Apply a Tone Model")
                .font(.headline)
            
            Text("Select a trained tone model and apply it to your DI recording.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            
            // Model picker
            Picker("Select Model", selection: $selectedModel) {
                Text("Choose a model...").tag(nil as ToneModel?)
                ForEach(api.models) { model in
                    Text(model.displayName).tag(model as ToneModel?)
                }
            }
            
            // DI file picker
            HStack {
                TextField("DI recording path...", text: $diPath)
                    .textFieldStyle(.roundedBorder)
                Button("Browse") {
                    let panel = NSOpenPanel()
                    panel.allowedContentTypes = [.audio]
                    panel.allowsMultipleSelection = false
                    if panel.runModal() == .OK, let url = panel.url {
                        diPath = url.path
                    }
                }
            }
            
            // Output path
            HStack {
                TextField("Output path...", text: $outputPath)
                    .textFieldStyle(.roundedBorder)
                Button("Browse") {
                    let panel = NSSavePanel()
                    panel.allowedContentTypes = [.audio]
                    panel.nameFieldStringValue = "tone_applied.wav"
                    if panel.runModal() == .OK, let url = panel.url {
                        outputPath = url.path
                    }
                }
            }
            
            Button("Apply Tone") {
                if let model = selectedModel {
                    api.applyTone(
                        modelId: model.id,
                        diPath: diPath,
                        outputPath: outputPath.isEmpty ? "output.wav" : outputPath
                    )
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(selectedModel == nil || diPath.isEmpty)
            .tint(.orange)
            
            Spacer()
        }
        .padding()
    }
}

// MARK: - Models View

struct ModelsView: View {
    @EnvironmentObject var api: ToneReplicatorAPI
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Trained Models")
                .font(.headline)
            
            if api.models.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "externaldrive.badge.plus")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text("No models yet")
                        .foregroundStyle(.secondary)
                    Text("Train your first tone model in the Train tab.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(api.models) { model in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(model.name)
                                .font(.headline)
                            Text("Source: \(model.sourceSong)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text("ESR: \(String(format: "%.4f", model.valEsr))")
                            .font(.monospaced)
                            .foregroundStyle(model.valEsr < 0.1 ? .green : .orange)
                    }
                }
            }
        }
        .padding()
    }
}