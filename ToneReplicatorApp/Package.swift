// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ToneReplicatorApp",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "ToneReplicatorApp", targets: ["ToneReplicatorApp"]),
    ],
    targets: [
        .executableTarget(
            name: "ToneReplicatorApp",
            path: "ToneReplicatorApp",
            exclude: ["Info.plist", "Entitlements.plist"]
        ),
    ]
)