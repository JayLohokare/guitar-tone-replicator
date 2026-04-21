import SwiftUI
import Combine
import AVFoundation
import UniformTypeIdentifiers

// MARK: - App

@main
struct ToneReplicatorApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var appState = AppState()
    
    var body: some Scene {
        Settings { EmptyView() }
    }
}

// MARK: - App Delegate

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: KeyWindow?
    @MainActor var appState = AppState()
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        
        let contentView = MainWindow()
            .environmentObject(appState)
            .frame(minWidth: 800, minHeight: 600)
        
        let hostingView = NSHostingView(rootView: contentView)
        hostingView.translatesAutoresizingMaskIntoConstraints = false
        
        let newWindow = KeyWindow()
        newWindow.contentView = hostingView
        newWindow.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        newWindow.title = "Tone Replicator"
        newWindow.titlebarAppearsTransparent = false
        newWindow.isReleasedWhenClosed = false
        newWindow.setFrame(NSRect(x: 0, y: 0, width: 900, height: 680), display: true)
        newWindow.center()
        newWindow.level = .normal
        newWindow.makeKeyAndOrderFront(nil)
        newWindow.orderFrontRegardless()
        NSApp.activate(ignoringOtherApps: true)
        
        self.window = newWindow
        appState.scanLocalModels()
    }
}

// MARK: - Key Window

class KeyWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
    
    override func keyDown(with event: NSEvent) {
        // Forward key events to the content view so SwiftUI text fields receive input
        contentView?.keyDown(with: event)
    }
}

// MARK: - Constants

let PROJECT_PATH = "/Users/jaylohokare/dev/guitar-tone-replicator"
let VENV_PYTHON = PROJECT_PATH + "/venv/bin/python"
let MODELS_DIR = "/Users/jaylohokare/ToneReplicator/models"
let SEPARATION_API = "http://localhost:8766"

// MARK: - App State

@MainActor
class AppState: ObservableObject {
    @Published var models: [LocalModel] = []
    @Published var selectedTab = 0
    
    func scanLocalModels() {
        let fm = FileManager.default
        var found: [LocalModel] = []
        
        guard let contents = try? fm.contentsOfDirectory(atPath: MODELS_DIR) else { return }
        
        for name in contents {
            let dir = MODELS_DIR + "/" + name
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: dir, isDirectory: &isDir), isDir.boolValue else { continue }
            
            let pthPath = dir + "/model.pth"
            let mlPath = dir + "/model.mlpackage"
            let metaPath = dir + "/metadata.json"
            
            guard fm.fileExists(atPath: pthPath) else { continue }
            
            var modelType = "wavenet"
            var modelSize = "standard"
            var bestEsr = -1.0
            var sampleRate = 48000
            
            if let data = try? Data(contentsOf: URL(fileURLWithPath: metaPath)),
               let meta = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                modelType = meta["model_type"] as? String ?? "wavenet"
                modelSize = meta["model_size"] as? String ?? "standard"
                bestEsr = meta["best_val_esr"] as? Double ?? -1.0
                sampleRate = meta["sample_rate"] as? Int ?? 48000
            }
            
            found.append(LocalModel(
                id: name,
                name: name.replacingOccurrences(of: "_", with: " "),
                path: dir,
                modelType: modelType,
                modelSize: modelSize,
                bestValEsr: bestEsr,
                sampleRate: sampleRate,
                hasCoreML: fm.fileExists(atPath: mlPath)
            ))
        }
        
        models = found.sorted { $0.name < $1.name }
    }
}

// MARK: - Local Model

struct LocalModel: Identifiable {
    let id: String
    let name: String
    let path: String
    let modelType: String
    let modelSize: String
    let bestValEsr: Double
    let sampleRate: Int
    let hasCoreML: Bool
}

// MARK: - Pipeline Job

@MainActor
class PipelineJob: ObservableObject {
    @Published var status: Status = .idle
    @Published var progress: Double = 0
    @Published var message: String = "Ready"
    @Published var currentStep: String = ""
    
    enum Status { case idle, downloading, separating, training, complete, error }
}

// MARK: - Main Window

struct MainWindow: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        VStack(spacing: 0) {
            HeaderBar()
            Divider()
            TabView(selection: $appState.selectedTab) {
                ReplicateView()
                    .tabItem { Label("Replicate", systemImage: "guitars") }
                    .tag(0)
                LivePlayView()
                    .tabItem { Label("Live Play", systemImage: "speaker.wave.2") }
                    .tag(1)
                ModelsView()
                    .tabItem { Label("Models", systemImage: "externaldrive") }
                    .tag(2)
            }
            .padding()
        }
    }
}

// MARK: - Header

struct HeaderBar: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "guitars")
                .font(.title2)
                .foregroundStyle(.orange)
            Text("Tone Replicator")
                .font(.title3.bold())
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
    }
}

// MARK: - Replicate View (No Server — Direct Process Calls)

struct ReplicateView: View {
    @EnvironmentObject var appState: AppState
    @State private var urlString = ""
    @State private var modelName = ""
    @State private var modelSize = "lite"
    @State private var epochs = 50
    @State private var startTimeText = ""
    @State private var endTimeText = ""
    @FocusState private var focusedField: Field?
    @ObservedObject private var job = PipelineJob()
    
    enum Field: Hashable {
        case url, modelName, startTime, endTime
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Replicate a Guitar Tone")
                    .font(.title2.bold())
                Text("Paste a song URL → download → extract guitar → train neural model → play it live")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            
            GroupBox("Song or Video URL") {
                VStack(alignment: .leading, spacing: 8) {
                    TextField("Paste YouTube, SoundCloud, or audio URL...", text: $urlString)
                        .textFieldStyle(.roundedBorder)
                        .focused($focusedField, equals: .url)
                        .onSubmit { focusedField = .modelName }
                    
                    HStack(spacing: 16) {
                        VStack(alignment: .leading) {
                            Text("Start (sec)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            TextField("0", text: $startTimeText)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 80)
                                .focused($focusedField, equals: .startTime)
                        }
                        VStack(alignment: .leading) {
                            Text("End (sec)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            TextField("end", text: $endTimeText)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 80)
                                .focused($focusedField, equals: .endTime)
                        }
                        Spacer()
                        Text("Leave blank for full song")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                }
                .padding(8)
            }
            
            GroupBox("Tone Model Settings") {
                HStack(spacing: 20) {
                    VStack(alignment: .leading) {
                        Text("Model Name")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        TextField("e.g. Metallica_Tone", text: $modelName)
                            .textFieldStyle(.roundedBorder)
                            .focused($focusedField, equals: .modelName)
                    }
                    VStack(alignment: .leading) {
                        Text("Size")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Picker("", selection: $modelSize) {
                            Text("Lite (fast)").tag("lite")
                            Text("Standard (best)").tag("standard")
                        }
                        .pickerStyle(.menu)
                    }
                    VStack(alignment: .leading) {
                        Text("Epochs")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Stepper("\(epochs)", value: $epochs, in: 10...500, step: 10)
                    }
                }
                .padding(8)
            }
            
            Button(action: startPipeline) {
                HStack(spacing: 8) {
                    Image(systemName: job.status == .idle ? "waveform.badge.magnifyingglass" : "arrow.trianglehead.2.clockwise")
                    Text(job.status == .idle ? "Replicate Tone" : job.status == .complete ? "Done!" : "Processing...")
                }
                .frame(maxWidth: .infinity)
                .font(.headline)
            }
            .buttonStyle(.borderedProminent)
            .tint(.orange)
            .disabled(urlString.isEmpty || modelName.isEmpty || (job.status != .idle && job.status != .complete && job.status != .error))
            
            if job.status != .idle {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(job.message)
                            .font(.subheadline)
                    }
                    ProgressView(value: job.progress)
                        .progressViewStyle(.linear)
                        .tint(.orange)
                    Text(job.currentStep)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
            
            if job.status == .complete {
                Button("🎸 Play it live →") {
                    appState.selectedTab = 1
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
            }
            
            Spacer()
        }
        .onAppear {
            // Auto-focus URL field when view appears so keyboard works immediately
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                focusedField = .url
            }
        }
    }
    
    private func startPipeline() {
        guard !urlString.isEmpty, !modelName.isEmpty else { return }
        
        job.status = .downloading
        job.message = "Starting pipeline..."
        job.progress = 0.01
        job.currentStep = "Launching Python process"
        
        let jobRef = job
        let appStateRef = appState
        
        let urlCopy = urlString
        let modelNameCopy = modelName
        let modelSizeCopy = modelSize
        let epochsCopy = epochs
        let startTimeCopy = startTimeText
        let endTimeCopy = endTimeText
        
        DispatchQueue.global(qos: .userInitiated).async {
            runPipelineDirect(
                url: urlCopy,
                modelName: modelNameCopy,
                modelSize: modelSizeCopy,
                epochs: epochsCopy,
                startTime: startTimeCopy,
                endTime: endTimeCopy,
                job: jobRef,
                appState: appStateRef
            )
        }
    }
}

// MARK: - Direct Pipeline (no server)

func runPipelineDirect(
    url: String,
    modelName: String,
    modelSize: String,
    epochs: Int,
    startTime: String,
    endTime: String,
    job: PipelineJob,
    appState: AppState
) {
    let tempDir = NSTemporaryDirectory() + "tone_replicator_\(modelName)"
    try? FileManager.default.createDirectory(atPath: tempDir, withIntermediateDirectories: true)
    
    // Step 1: Download with yt-dlp
    DispatchQueue.main.async {
        job.status = .downloading
        job.message = "Downloading audio..."
        job.currentStep = "Running yt-dlp"
        job.progress = 0.05
    }
    
    let audioPath = tempDir + "/downloaded.wav"
    var ytArgs = [url, "-x", "--audio-format", "wav", "-o", audioPath]
    if !startTime.isEmpty { ytArgs += ["--download-section", "*\(startTime)-\(endTime.isEmpty ? "inf" : endTime)"] }
    
    let ytProc = Process()
    ytProc.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/yt-dlp")
    ytProc.arguments = ytArgs
    ytProc.currentDirectoryURL = URL(fileURLWithPath: tempDir)
    let ytPipe = Pipe()
    ytProc.standardOutput = ytPipe
    ytProc.standardError = ytPipe
    
    do {
        try ytProc.run()
        ytProc.waitUntilExit()
    } catch {
        DispatchQueue.main.async {
            job.status = .error
            job.message = "yt-dlp failed: \(error.localizedDescription)"
        }
        return
    }
    
    if ytProc.terminationStatus != 0 {
        let output = String(data: ytPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        DispatchQueue.main.async {
            job.status = .error
            job.message = "Download failed: \(output.prefix(300))"
        }
        return
    }
    
    // Find actual downloaded file (yt-dlp may add extension)
    let fm = FileManager.default
    var actualAudioPath = audioPath
    if !fm.fileExists(atPath: audioPath) {
        // Look for any audio file in temp dir
        if let files = try? fm.contentsOfDirectory(atPath: tempDir) {
            for f in files {
                if f.hasSuffix(".wav") || f.hasSuffix(".mp3") || f.hasSuffix(".m4a") || f.hasSuffix(".webm") {
                    actualAudioPath = tempDir + "/" + f
                    break
                }
            }
        }
    }
    
    // Step 2: Separate guitar (via separation API on port 8766)
    DispatchQueue.main.async {
        job.status = .separating
        job.message = "Extracting guitar stem..."
        job.currentStep = "Sending to Demucs MLX"
        job.progress = 0.15
    }
    
    // Check if separation API is available
    let separateURL = URL(string: SEPARATION_API)!
    var separationAvailable = false
    let sem = DispatchSemaphore(value: 0)
    URLSession.shared.dataTask(with: separateURL) { data, response, error in
        separationAvailable = (response as? HTTPURLResponse)?.statusCode == 200
        sem.signal()
    }.resume()
    _ = sem.wait(timeout: .now() + 5)
    
    var guitarPath = actualAudioPath
    if separationAvailable {
        // Upload to separation API
        guard let audioData = try? Data(contentsOf: URL(fileURLWithPath: actualAudioPath)) else {
            DispatchQueue.main.async {
                job.status = .error
                job.message = "Failed to read downloaded audio"
            }
            return
        }
        
        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: URL(string: SEPARATION_API + "/separate")!)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        body.append(audioData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body
        
        let sepSem = DispatchSemaphore(value: 0)
        var jobId: String?
        
        URLSession.shared.dataTask(with: request) { data, _, _ in
            if let data = data, let result = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                jobId = result["job_id"] as? String
            }
            sepSem.signal()
        }.resume()
        _ = sepSem.wait(timeout: .now() + 30)
        
        if let jId = jobId {
            // Poll for completion
            var done = false
            while !done {
                Thread.sleep(forTimeInterval: 2)
                let pollSem = DispatchSemaphore(value: 0)
                var statusVal = ""
                URLSession.shared.dataTask(with: URL(string: SEPARATION_API + "/status/\(jId)")!) { data, _, _ in
                    if let data = data, let result = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        statusVal = result["status"] as? String ?? ""
                    }
                    pollSem.signal()
                }.resume()
                _ = pollSem.wait(timeout: .now() + 10)
                
                if statusVal == "completed" {
                    // Download the result
                    let dlSem = DispatchSemaphore(value: 0)
                    let outputPath = tempDir + "/guitar_stem.wav"
                    URLSession.shared.dataTask(with: URL(string: SEPARATION_API + "/download/\(jId)")!) { data, _, _ in
                        if let data = data { try? data.write(to: URL(fileURLWithPath: outputPath)) }
                        dlSem.signal()
                    }.resume()
                    _ = dlSem.wait(timeout: .now() + 30)
                    
                    if fm.fileExists(atPath: outputPath) {
                        guitarPath = outputPath
                    }
                    done = true
                } else if statusVal == "failed" {
                    done = true
                }
                
                DispatchQueue.main.async {
                    job.progress = 0.15 + 0.15
                }
            }
        }
    } else {
        DispatchQueue.main.async {
            job.message = "Separation API offline — using full audio"
        }
    }
    
    // Step 3: Train model
    DispatchQueue.main.async {
        job.status = .training
        job.message = "Training tone model..."
        job.currentStep = "Running PyTorch training"
        job.progress = 0.4
    }
    
    let trainScript = """
    import sys
    sys.path.insert(0, '\(PROJECT_PATH)')
    from src.core.trainer import ToneTrainer
    from src.separation.separator import prepare_target_tone
    from pathlib import Path
    
    # Prepare target tone
    prepared = prepare_target_tone('\(guitarPath)')
    
    # Train
    trainer = ToneTrainer(model_type='wavenet', model_size='\(modelSize)')
    
    def progress_cb(entry):
        epoch = entry.get('epoch', 0)
        val_esr = entry.get('val_esr', 1.0)
        print(f'PROGRESS:{epoch}:\(epochs):{val_esr:.6f}', flush=True)
    
    result = trainer.train_blind(
        target_tone_path=prepared,
        epochs=\(epochs),
        save_path='\(MODELS_DIR)/\(modelName)',
        progress_callback=progress_cb,
    )
    print(f'DONE:{result["best_val_esr"]:.6f}', flush=True)
    """
    
    let trainProc = Process()
    trainProc.executableURL = URL(fileURLWithPath: VENV_PYTHON)
    trainProc.arguments = ["-c", trainScript]
    trainProc.currentDirectoryURL = URL(fileURLWithPath: PROJECT_PATH)
    
    let trainOut = Pipe()
    let trainErr = Pipe()
    trainProc.standardOutput = trainOut
    trainProc.standardError = trainErr
    
    // Read stdout line by line for progress
    trainOut.fileHandleForReading.readabilityHandler = { handle in
        let data = handle.availableData
        guard let output = String(data: data, encoding: .utf8) else { return }
        
        for line in output.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("PROGRESS:") {
                let parts = trimmed.replacingOccurrences(of: "PROGRESS:", with: "").components(separatedBy: ":")
                if parts.count == 3,
                   let epoch = Int(parts[0]),
                   let total = Int(parts[1]),
                   let esr = Double(parts[2]) {
                    DispatchQueue.main.async {
                        job.progress = 0.4 + (Double(epoch) / Double(total)) * 0.55
                        job.message = "Epoch \(epoch)/\(total): ESR = \(String(format: "%.4f", esr))"
                        job.currentStep = "Training on \(modelSize) model"
                    }
                }
            } else if trimmed.hasPrefix("DONE:") {
                let esrStr = trimmed.replacingOccurrences(of: "DONE:", with: "")
                DispatchQueue.main.async {
                    job.message = "Training complete! ESR = \(esrStr)"
                }
            }
        }
    }
    
    do {
        try trainProc.run()
        trainProc.waitUntilExit()
    } catch {
        DispatchQueue.main.async {
            job.status = .error
            job.message = "Training failed: \(error.localizedDescription)"
        }
        return
    }
    
    if trainProc.terminationStatus != 0 {
        let errData = trainErr.fileHandleForReading.readDataToEndOfFile()
        let errMsg = String(data: errData, encoding: .utf8) ?? "unknown"
        DispatchQueue.main.async {
            job.status = .error
            job.message = "Training error: \(errMsg.prefix(300))"
        }
        return
    }
    
    DispatchQueue.main.async {
        job.status = .complete
        job.message = "Tone replicated!"
        job.currentStep = "Switch to Live Play tab to try it"
        job.progress = 1.0
        appState.scanLocalModels()
    }
}

// MARK: - Live Play View

struct LivePlayView: View {
    @EnvironmentObject var appState: AppState
    @StateObject private var engine = LiveAudioEngine()
    @State private var selectedModelId: String?
    @State private var statusMessage = "Select a model and hit Play"
    @State private var isModelLoaded = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Live Play")
                    .font(.title2.bold())
                Text("Play your guitar through the trained tone model in real-time. Connect your audio interface (e.g. Scarlett Solo) and hit Play.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            
            GroupBox("Tone Model") {
                HStack(spacing: 16) {
                    Picker("Model", selection: $selectedModelId) {
                        Text("Choose a model...").tag(nil as String?)
                        ForEach(appState.models) { model in
                            HStack {
                                Text(model.name)
                                if model.hasCoreML {
                                    Image(systemName: "apple.logo")
                                        .foregroundStyle(.green)
                                        .font(.caption)
                                }
                            }
                            .tag(model.id as String?)
                        }
                    }
                    .pickerStyle(.menu)
                    
                    Button("Load") {
                        loadSelectedModel()
                    }
                    .buttonStyle(.bordered)
                    .disabled(selectedModelId == nil)
                    
                    if isModelLoaded {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    }
                }
                .padding(8)
            }
            
            GroupBox("Controls") {
                VStack(spacing: 16) {
                    HStack {
                        Text("Input Gain")
                            .frame(width: 100, alignment: .leading)
                        Slider(value: $engine.inputGainDb, in: -40...40)
                        Text("\(engine.inputGainDb, specifier: "%.1f") dB")
                            .font(.caption)
                            .monospacedDigit()
                            .frame(width: 60)
                    }
                    
                    HStack {
                        Text("Output Gain")
                            .frame(width: 100, alignment: .leading)
                        Slider(value: $engine.outputGainDb, in: -40...40)
                        Text("\(engine.outputGainDb, specifier: "%.1f") dB")
                            .font(.caption)
                            .monospacedDigit()
                            .frame(width: 60)
                    }
                    
                    HStack {
                        Text("Dry/Wet")
                            .frame(width: 100, alignment: .leading)
                        Slider(value: $engine.dryWet, in: 0...100)
                        Text("\(engine.dryWet, specifier: "%.0f")%")
                            .font(.caption)
                            .monospacedDigit()
                            .frame(width: 60)
                    }
                }
                .padding(8)
            }
            
            GroupBox("Levels") {
                HStack(spacing: 24) {
                    VStack(spacing: 4) {
                        Text("IN")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Color.gray.opacity(0.2))
                            .overlay(alignment: .bottom) {
                                RoundedRectangle(cornerRadius: 2)
                                    .fill(LinearGradient(colors: [.green, .yellow, .red], startPoint: .bottom, endPoint: .top))
                                    .frame(height: CGFloat(engine.inputLevel) * 120)
                            }
                            .frame(width: 20, height: 120)
                    }
                    
                    VStack(spacing: 4) {
                        Text("OUT")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Color.gray.opacity(0.2))
                            .overlay(alignment: .bottom) {
                                RoundedRectangle(cornerRadius: 2)
                                    .fill(LinearGradient(colors: [.green, .yellow, .red], startPoint: .bottom, endPoint: .top))
                                    .frame(height: CGFloat(engine.outputLevel) * 120)
                            }
                            .frame(width: 20, height: 120)
                    }
                    
                    Spacer()
                    
                    VStack(spacing: 8) {
                        Button {
                            if engine.isRunning { engine.stop() } else { engine.start() }
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: engine.isRunning ? "stop.fill" : "play.fill")
                                Text(engine.isRunning ? "Stop" : "Play")
                            }
                            .frame(width: 100)
                            .font(.headline)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(engine.isRunning ? .red : .green)
                        
                        Toggle("Bypass", isOn: $engine.bypass)
                            .toggleStyle(.switch)
                            .controlSize(.small)
                    }
                }
                .padding(8)
            }
            .onChange(of: engine.inputGainDb) { engine.syncParams() }
            .onChange(of: engine.outputGainDb) { engine.syncParams() }
            .onChange(of: engine.dryWet) { engine.syncParams() }
            .onChange(of: engine.bypass) { engine.syncParams() }
            
            HStack {
                Circle()
                    .fill(engine.isRunning ? Color.green : Color.red)
                    .frame(width: 8, height: 8)
                Text(engine.statusText.isEmpty ? statusMessage : engine.statusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if engine.isRunning {
                    Text("Latency: \(engine.latencyMs, specifier: "%.1f") ms")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            
            Spacer()
        }
    }
    
    private func loadSelectedModel() {
        guard let id = selectedModelId,
              let model = appState.models.first(where: { $0.id == id }) else { return }
        
        statusMessage = "Loading \(model.name)..."
        engine.loadModel(path: model.path) { success in
            isModelLoaded = success
            statusMessage = success ? "Model loaded — hit Play!" : "Failed to load model"
        }
    }
}

// MARK: - Live Audio Engine
// Thread-safe design: audio tap only touches locked params + raw level vars.
// Published properties are updated from main thread via Timer.
// Real-time processing uses a ring buffer to decouple audio thread from inference.

final class LiveAudioEngine: ObservableObject, @unchecked Sendable {
    @Published var isRunning = false
    @Published var inputGainDb: Float = 0.0
    @Published var outputGainDb: Float = 0.0
    @Published var dryWet: Float = 100.0
    @Published var bypass = false
    @Published var inputLevel: Float = 0.0
    @Published var outputLevel: Float = 0.0
    @Published var latencyMs: Float = 0.0
    @Published var statusText: String = "Ready"
    
    // Thread-safe parameter copies (written by main thread, read by audio thread)
    private let paramsLock = NSLock()
    private var safeInputGain: Float = 0.0
    private var safeOutputGain: Float = 0.0
    private var safeDryWet: Float = 100.0
    private var safeBypass: Bool = false
    
    // Raw level values (written by audio thread, read by main thread via timer)
    private var rawInputLevel: Float = 0
    private var rawOutputLevel: Float = 0
    
    private var audioEngine: AVAudioEngine?
    private var inferenceProcess: Process?
    private var inferenceStdin: FileHandle?
    private var inferenceStdout: FileHandle?
    private var isModelLoaded = false
    private var levelTimer: Timer?
    
    // Ring buffer for decoupling audio thread from inference
    private let ringBufferSize = 16384  // ~370ms at 44100Hz
    private var inputRingBuffer = [Float](repeating: 0, count: 16384)
    private var outputRingBuffer = [Float](repeating: 0, count: 16384)
    private var ringWriteIdx = 0
    private var ringReadIdx = 0
    private var ringCount = 0
    private let ringLock = NSLock()
    
    // Inference thread
    private var inferenceThread: Thread?
    private var inferenceRunning = false
    private let inferenceChunkSize = 2048  // Process in larger chunks for efficiency
    private var bypassPending = false
    
    // Output buffer for the audio tap
    private var processedOutputBuffer = [Float]()
    private var processedReadIdx = 0
    private let outputLock = NSLock()
    
    func start() {
        if isRunning {
            print("[Audio] Already running, stopping first...")
            stop()
        }
        
        // Sync params before starting
        syncParams()
        
        print("[Audio] Starting audio engine...")
        let engine = AVAudioEngine()
        self.audioEngine = engine
        
        let inputNode = engine.inputNode
        let outputNode = engine.outputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)
        
        print("[Audio] Hardware format: sr=\(hwFormat.sampleRate), channels=\(hwFormat.channelCount), isInterleaved=\(hwFormat.isInterleaved)")
        
        if hwFormat.sampleRate == 0 || hwFormat.channelCount == 0 {
            statusText = "Error: No audio input found. Connect an audio interface or mic."
            print("[Audio] No audio input device detected!")
            self.audioEngine = nil
            return
        }
        
        // Use the hardware's native format for the tap to avoid format conversion issues
        // The Scarlett Solo reports 2 channels at 44100Hz
        let tapFormat = AVAudioFormat(standardFormatWithSampleRate: hwFormat.sampleRate, channels: 1)!
        
        do {
            try inputNode.installTap(onBus: 0, bufferSize: 1024, format: tapFormat) { [weak self] buffer, _ in
                self?.processAudioBuffer(buffer)
            }
            print("[Audio] Input tap installed successfully")
        } catch {
            statusText = "Error installing audio tap: \(error.localizedDescription)"
            print("[Audio] Failed to install tap: \(error)")
            self.audioEngine = nil
            return
        }
        
        // Connect input → mainMixer → output so audio passes through
        // The tap intercepts for processing but we still need the connection for output
        engine.connect(inputNode, to: engine.mainMixerNode, format: hwFormat)
        engine.connect(engine.mainMixerNode, to: outputNode, format: outputNode.outputFormat(forBus: 0))
        
        // Reset ring buffers
        ringWriteIdx = 0
        ringReadIdx = 0
        ringCount = 0
        processedOutputBuffer.removeAll()
        processedReadIdx = 0
        
        // Start inference thread if model is loaded
        if isModelLoaded {
            startInferenceThread()
        }
        
        do {
            try engine.start()
            isRunning = true
            let sr = Int(hwFormat.sampleRate)
            let ch = Int(hwFormat.channelCount)
            statusText = "Running: \(sr)Hz / \(ch)ch input"
            startLevelTimer()
            print("[Audio] Engine started successfully")
        } catch {
            statusText = "Error: \(error.localizedDescription)"
            print("[Audio] Engine start failed: \(error)")
        }
    }
    
    func stop() {
        guard isRunning else { return }
        inferenceRunning = false
        inferenceThread = nil
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        isRunning = false
        inputLevel = 0
        outputLevel = 0
        statusText = "Stopped"
        stopLevelTimer()
    }
    
    /// Copy @Published params to thread-safe storage for audio thread
    func syncParams() {
        paramsLock.lock()
        safeInputGain = inputGainDb
        safeOutputGain = outputGainDb
        safeDryWet = dryWet
        safeBypass = bypass
        paramsLock.unlock()
        
        // Also send param updates to inference process
        sendParameterUpdates()
    }
    
    func loadModel(path: String, completion: @Sendable @escaping (Bool) -> Void) {
        if inferenceProcess == nil {
            launchInferenceProcess()
        }
        
        guard let stdin = inferenceStdin else {
            DispatchQueue.main.async { completion(false) }
            return
        }
        
        // First, drain any startup messages from the inference process
        // The process sends a {"status": "ready"} message on startup
        // We need to consume it before sending our load command
        if let stdout = inferenceStdout {
            let existingData = stdout.availableData
            if !existingData.isEmpty {
                print("[Inference] Drained startup data: \(String(data: existingData, encoding: .utf8) ?? "(binary)")")
            }
        }
        
        let cmd = "{\"cmd\": \"load\", \"model_path\": \"\(path)\"}\n"
        guard let data = cmd.data(using: .utf8) else {
            DispatchQueue.main.async { completion(false) }
            return
        }
        stdin.write(data)
        
        // Read response with timeout on background thread
        DispatchQueue.global(qos: .userInitiated).async {
            guard let stdout = self.inferenceStdout else {
                DispatchQueue.main.async { completion(false) }
                return
            }
            
            // Read until we get a JSON response line
            var response = ""
            let startTime = Date()
            let timeout: TimeInterval = 15.0  // Model loading can take a few seconds
            
            while Date().timeIntervalSince(startTime) < timeout {
                let availableData = stdout.availableData
                if !availableData.isEmpty {
                    if let str = String(data: availableData, encoding: .utf8) {
                        response += str
                        // Check if we have a complete JSON line
                        if let newlineIdx = response.firstIndex(of: "\n") {
                            let jsonLine = String(response[..<newlineIdx]).trimmingCharacters(in: .whitespaces)
                            if let jsonData = jsonLine.data(using: .utf8),
                               let result = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {
                                let isLoaded = result["status"] as? String == "ok"
                                DispatchQueue.main.async {
                                    self.isModelLoaded = isLoaded
                                    self.statusText = isLoaded ? "Model loaded — hit Play!" : "Failed to load model: \(result["message"] as? String ?? "unknown")"
                                    completion(isLoaded)
                                }
                                return
                            }
                        }
                    }
                }
                Thread.sleep(forTimeInterval: 0.1)
            }
            
            // Timeout
            DispatchQueue.main.async {
                self.statusText = "Timeout loading model"
                completion(false)
            }
        }
    }
    
    private func launchInferenceProcess() {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: VENV_PYTHON)
        proc.arguments = [PROJECT_PATH + "/realtime_inference.py"]
        proc.currentDirectoryURL = URL(fileURLWithPath: PROJECT_PATH)
        
        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        proc.standardInput = stdinPipe
        proc.standardOutput = stdoutPipe
        proc.standardError = stderrPipe
        
        inferenceStdin = stdinPipe.fileHandleForWriting
        inferenceStdout = stdoutPipe.fileHandleForReading
        
        // Log stderr for debugging
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let str = String(data: data, encoding: .utf8) {
                print("[Inference stderr] \(str.trimmingCharacters(in: .whitespacesAndNewlines))")
            }
        }
        
        do {
            try proc.run()
            inferenceProcess = proc
            statusText = "Inference process launched (PID: \(proc.processIdentifier))"
            print("[Inference] Process launched, PID: \(proc.processIdentifier)")
        } catch {
            statusText = "Failed to launch inference: \(error.localizedDescription)"
            print("[Inference] Failed to launch: \(error)")
        }
    }
    
    // MARK: - Inference Thread
    
    private func startInferenceThread() {
        guard isModelLoaded else { return }
        inferenceRunning = true
        
        let thread = Thread {
            [weak self] in
            self?.inferenceLoop()
        }
        thread.qualityOfService = .userInteractive
        thread.start()
        inferenceThread = thread
        print("[Inference] Background thread started")
    }
    
    private func inferenceLoop() {
        var chunkBuffer = [Float](repeating: 0, count: inferenceChunkSize)
        
        while inferenceRunning {
            // Try to get a chunk from the ring buffer
            ringLock.lock()
            let available = ringCount
            let framesToProcess = min(available, inferenceChunkSize)
            
            if framesToProcess == 0 {
                ringLock.unlock()
                Thread.sleep(forTimeInterval: 0.002)  // 2ms poll
                continue
            }
            
            // Copy from ring buffer
            for i in 0..<framesToProcess {
                chunkBuffer[i] = inputRingBuffer[ringReadIdx]
                ringReadIdx = (ringReadIdx + 1) % ringBufferSize
            }
            ringCount -= framesToProcess
            ringLock.unlock()
            
            // Send to inference process
            guard let stdin = inferenceStdin else { continue }
            
            let cmd = "{\"cmd\": \"process\", \"sample_rate\": 44100, \"channels\": 1, \"frames\": \(framesToProcess)}\n"
            guard let cmdData = cmd.data(using: .utf8) else { continue }
            
            // Send JSON command
            stdin.write(cmdData)
            
            // Send raw audio data
            var audioData = Data(capacity: framesToProcess * 4)
            for i in 0..<framesToProcess {
                audioData.withUnsafeMutableBytes { ptr in
                    ptr.baseAddress!.assumingMemoryBound(to: Float.self)[i] = chunkBuffer[i]
                }
            }
            // Actually, just use the float array directly
            chunkBuffer.withUnsafeBufferPointer { ptr in
                audioData.append(UnsafeBufferPointer(start: ptr.baseAddress, count: framesToProcess))
            }
            stdin.write(audioData)
            
            // Read response JSON
            guard let stdout = inferenceStdout else { continue }
            var responseStr = ""
            let startTime = Date()
            
            while Date().timeIntervalSince(startTime) < 0.5 {
                let data = stdout.availableData
                if !data.isEmpty, let str = String(data: data, encoding: .utf8) {
                    responseStr += str
                    if let newlineIdx = responseStr.firstIndex(of: "\n") {
                        let jsonLine = String(responseStr[..<newlineIdx]).trimmingCharacters(in: .whitespaces)
                        // Check if it's a valid JSON
                        if let jsonData = jsonLine.data(using: .utf8),
                           let result = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {
                            // It's our response
                            if result["status"] as? String == "ok",
                               let outFrames = result["frames"] as? Int {
                                // Read the raw audio output
                                let bytesNeeded = outFrames * 4
                                var outputAudio = Data()
                                var readAttempts = 0
                                while outputAudio.count < bytesNeeded && readAttempts < 100 {
                                    let moreData = stdout.availableData
                                    if !moreData.isEmpty {
                                        outputAudio.append(moreData)
                                    } else {
                                        Thread.sleep(forTimeInterval: 0.001)
                                    }
                                    readAttempts += 1
                                }
                                
                                if outputAudio.count >= bytesNeeded {
                                    // Convert to float array and push to output buffer
                                    outputAudio.withUnsafeBytes { ptr in
                                        let floatPtr = ptr.baseAddress!.assumingMemoryBound(to: Float.self)
                                        let outputFloats = Array(UnsafeBufferPointer(start: floatPtr, count: outFrames))
                                        
                                        outputLock.lock()
                                        processedOutputBuffer.append(contentsOf: outputFloats)
                                        outputLock.unlock()
                                    }
                                }
                            }
                            break
                        }
                    }
                }
                Thread.sleep(forTimeInterval: 0.001)
            }
        }
        print("[Inference] Background thread ended")
    }
    
    // MARK: - Audio Processing (called from real-time audio thread)
    
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return }
        
        // Read thread-safe params (no MainActor access!)
        paramsLock.lock()
        let inputGain = safeInputGain
        let outputGain = safeOutputGain
        let wet = safeDryWet / 100.0
        let dry = 1.0 - wet
        let isBypass = safeBypass
        paramsLock.unlock()
        
        let inputGainLinear = Float(pow(10.0, Double(inputGain) * 0.05))
        let outputGainLinear = Float(pow(10.0, Double(outputGain) * 0.05))
        
        // Compute input RMS
        var inputRms: Float = 0
        for i in 0..<frameCount {
            inputRms += channelData[i] * channelData[i]
        }
        inputRms = sqrt(inputRms / Float(frameCount))
        rawInputLevel = min(inputRms * 3.0, 1.0)
        
        // Log audio flow
        if rawInputLevel > 0.001 {
            print("[Audio] Buffer: \(frameCount) frames, rms=\(inputRms), peak=\(abs(channelData[0]))")
        }
        
        if isBypass || !isModelLoaded {
            // Passthrough with gain
            for i in 0..<frameCount {
                channelData[i] *= outputGainLinear
            }
            rawOutputLevel = min(inputRms * 3.0 * outputGainLinear, 1.0)
        } else {
            // Push input to ring buffer for inference thread
            ringLock.lock()
            for i in 0..<frameCount {
                if ringCount < ringBufferSize {
                    inputRingBuffer[ringWriteIdx] = channelData[i] * inputGainLinear
                    ringWriteIdx = (ringWriteIdx + 1) % ringBufferSize
                    ringCount += 1
                }
            }
            ringLock.unlock()
            
            // Pull processed output from inference thread
            outputLock.lock()
            let availableOutput = min(processedOutputBuffer.count - processedReadIdx, frameCount)
            if availableOutput > 0 {
                for i in 0..<availableOutput {
                    let wetSignal = processedOutputBuffer[processedReadIdx + i] * outputGainLinear
                    let drySignal = channelData[i] * inputGainLinear * outputGainLinear
                    channelData[i] = drySignal * dry + wetSignal * wet
                }
                processedReadIdx += availableOutput
                
                // If we read all available output, reset buffer
                if processedReadIdx >= processedOutputBuffer.count {
                    processedOutputBuffer.removeAll(keepingCapacity: true)
                    processedReadIdx = 0
                }
                
                // Fill remaining frames with dry signal if not enough processed output
                for i in availableOutput..<frameCount {
                    let signal = channelData[i] * inputGainLinear * outputGainLinear
                    channelData[i] = signal
                }
                
                var outputRms: Float = 0
                for i in 0..<frameCount {
                    outputRms += channelData[i] * channelData[i]
                }
                rawOutputLevel = min(sqrt(outputRms / Float(frameCount)) * 3.0, 1.0)
            } else {
                // No processed output yet - passthrough with gain
                for i in 0..<frameCount {
                    channelData[i] *= outputGainLinear
                }
                rawOutputLevel = min(inputRms * 3.0 * outputGainLinear, 1.0)
            }
            outputLock.unlock()
        }
    }
    
    // MARK: - Level Meter Timer (polls raw values on main thread)
    
    private func startLevelTimer() {
        levelTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            self.inputLevel = self.rawInputLevel
            self.outputLevel = self.rawOutputLevel
        }
    }
    
    private func stopLevelTimer() {
        levelTimer?.invalidate()
        levelTimer = nil
    }
    
    func sendParameterUpdates() {
        guard isModelLoaded, let stdin = inferenceStdin else { return }
        paramsLock.lock()
        let ig = safeInputGain
        let og = safeOutputGain
        let dw = safeDryWet
        paramsLock.unlock()
        let cmd = "{\"cmd\": \"set_param\", \"input_gain_db\": \(ig), \"output_gain_db\": \(og), \"dry_wet\": \(dw)}\n"
        if let data = cmd.data(using: .utf8) {
            stdin.write(data)
        }
    }
}



// MARK: - Models View

struct ModelsView: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Trained Models")
                    .font(.title2.bold())
                Spacer()
                Button("Refresh") { appState.scanLocalModels() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
            
            if appState.models.isEmpty {
                VStack(spacing: 12) {
                    Spacer()
                    Image(systemName: "externaldrive.badge.plus")
                        .font(.system(size: 48))
                        .foregroundStyle(.secondary)
                    Text("No models yet")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                    Text("Use the Replicate tab to create your first tone model.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                    Spacer()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                Table(appState.models) {
                    TableColumn("Name") { model in Text(model.name).fontWeight(.medium) }
                    TableColumn("Type") { model in Text("\(model.modelType)/\(model.modelSize)").font(.caption) }
                    TableColumn("ESR") { model in
                        if model.bestValEsr >= 0 {
                            Text(String(format: "%.4f", model.bestValEsr))
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(model.bestValEsr < 0.1 ? .green : .orange)
                        } else {
                            Text("—").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    TableColumn("CoreML") { model in
                        if model.hasCoreML {
                            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                        } else {
                            Image(systemName: "xmark.circle").foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .padding()
    }
}