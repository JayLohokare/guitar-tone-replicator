// swift-tools-version: 5.9
// Tone Replicator Plugin - Swift Package
// MIT License

import PackageDescription

let package = Package(
    name: "ToneReplicatorPlugin",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "ToneReplicatorApp",
            targets: ["ToneReplicatorApp"]
        ),
    ],
    targets: [
        // C++ DSP kernel library
        .target(
            name: "ToneReplicatorDSP",
            path: "Sources/ToneReplicatorDSP",
            sources: [
                "ToneReplicatorKernel.cpp",
                "ToneReplicatorKernelBridge.cpp",
            ],
            publicHeadersPath: ".",
            cxxSettings: [
                .define("SWIFT_PACKAGE", to: "1"),
            ],
            linkerSettings: [
                .linkedLibrary("c++"),
            ]
        ),
        // Standalone app (Swift only, calls C++ via bridging layer)
        .executableTarget(
            name: "ToneReplicatorApp",
            dependencies: [
                "ToneReplicatorDSP",
            ],
            path: "Sources/ToneReplicatorApp",
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"]),
            ],
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreAudio"),
                .linkedFramework("CoreML"),
                .linkedFramework("AudioToolbox"),
            ]
        ),
    ],
    cLanguageStandard: .c17,
    cxxLanguageStandard: .cxx17
)