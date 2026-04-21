// ModelStore.swift
// Manages available tone models and model selection
// MIT License

import Foundation
import Combine

struct ToneModel: Identifiable, Hashable {
    let id: UUID
    let name: String
    let url: URL
    let modelType: String  // "wavenet" or "lstm"
    let modelSize: String  // "nano", "lite", "standard"
    let sampleRate: Double
    let esr: Double?

    var description: String {
        "\(modelType)/\(modelSize) · \(Int(sampleRate))Hz" + (esr.map { " · ESR: \(String(format: "%.4f", $0))" } ?? "")
    }

    static func == (lhs: ToneModel, rhs: ToneModel) -> Bool {
        lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

class ModelStore: ObservableObject {
    @Published var models: [ToneModel] = []
    @Published var activeModel: ToneModel?
    @Published var selectedModelID: UUID?

    private let modelsDirectory: URL
    private let fileManager = FileManager.default

    init() {
        modelsDirectory = fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent("ToneReplicator")
            .appendingPathComponent("models")
        refreshModels()
    }

    func refreshModels() {
        var discoveredModels: [ToneModel] = []

        guard let enumerator = fileManager.enumerator(at: modelsDirectory,
                                                       includingPropertiesForKeys: nil,
                                                       options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]) else {
            return
        }

        for case let directoryURL as URL in enumerator {
            guard directoryURL.hasDirectoryPath else { continue }

            let dirName = directoryURL.lastPathComponent
            let metadataPath = directoryURL.appendingPathComponent("metadata.json").path
            let mlpackagePath = directoryURL.appendingPathComponent("model.mlpackage").path
            let mlmodelcPath = directoryURL.appendingPathComponent("model.mlmodelc").path
            let pthPath = directoryURL.appendingPathComponent("model.pth").path

            // Check if this directory contains a model
            let hasPth = fileManager.fileExists(atPath: pthPath)
            let hasMLPackage = fileManager.fileExists(atPath: mlpackagePath)
            let hasMLModelC = fileManager.fileExists(atPath: mlmodelcPath)

            if !hasPth && !hasMLPackage && !hasMLModelC {
                continue
            }

            // Load metadata if available
            var modelType = "wavenet"
            var modelSize = "lite"
            var sampleRate = 44100.0
            var esr: Double? = nil

            if fileManager.fileExists(atPath: metadataPath),
               let data = fileManager.contents(atPath: metadataPath),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                modelType = json["model_type"] as? String ?? modelType
                modelSize = json["model_size"] as? String ?? modelSize
                sampleRate = json["sample_rate"] as? Double ?? sampleRate
                esr = json["best_val_esr"] as? Double
            }

            let modelURL: URL
            if hasMLPackage {
                modelURL = directoryURL.appendingPathComponent("model.mlpackage")
            } else if hasMLModelC {
                modelURL = directoryURL.appendingPathComponent("model.mlmodelc")
            } else {
                modelURL = directoryURL.appendingPathComponent("model.pth")
            }

            let model = ToneModel(
                id: UUID(),
                name: dirName.replacingOccurrences(of: "_", with: " "),
                url: modelURL,
                modelType: modelType,
                modelSize: modelSize,
                sampleRate: sampleRate,
                esr: esr
            )
            discoveredModels.append(model)
        }

        // Sort by name
        discoveredModels.sort { $0.name < $1.name }

        DispatchQueue.main.async {
            self.models = discoveredModels
        }
    }

    func importModel(from url: URL) {
        // Copy model file to models directory
        let fileName = url.lastPathComponent
        let dirName = url.deletingPathExtension().lastPathComponent
        let destDir = modelsDirectory.appendingPathComponent(dirName)

        do {
            try fileManager.createDirectory(at: destDir, withIntermediateDirectories: true)
            let destURL = destDir.appendingPathComponent(fileName)

            if fileManager.fileExists(atPath: destURL.path) {
                try fileManager.removeItem(at: destURL)
            }
            try fileManager.copyItem(at: url, to: destURL)
            refreshModels()
        } catch {
            print("Failed to import model: \(error)")
        }
    }
}