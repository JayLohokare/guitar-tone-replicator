// ToneReplicatorPluginApp.swift
// Standalone app for Tone Replicator AUv3 plugin
// MIT License

import SwiftUI

@main
struct ToneReplicatorPluginApp: App {
    @StateObject private var modelStore = ModelStore()
    @StateObject private var audioEngine = AudioEngineManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(modelStore)
                .environmentObject(audioEngine)
                .frame(minWidth: 600, minHeight: 500)
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified)
        .defaultSize(width: 700, height: 600)
    }
}