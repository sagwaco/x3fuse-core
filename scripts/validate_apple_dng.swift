#!/usr/bin/env swift
// Decode actual RAW pixels with Apple's engine; never modifies the input DNGs.
// Example: swift scripts/validate_apple_dng.swift --output /tmp/apple-check \
//   --scale 1 --edits photo.dng
// --decoder accepts an exact supportedDecoderVersions rawValue from report.json.

import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

struct ValidationError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

struct Options {
    var output: URL
    var decoder: String?
    var scale: Float = 0.125
    var edits = false
    var inputs: [URL]
}

func parseOptions() throws -> Options {
    let arguments = Array(CommandLine.arguments.dropFirst())
    var output: URL?
    var decoder: String?
    var scale: Float = 0.125
    var edits = false
    var inputs: [URL] = []
    var index = 0
    while index < arguments.count {
        let argument = arguments[index]
        switch argument {
        case "--help", "-h":
            print("""
            Usage: swift validate_apple_dng.swift --output DIR [--decoder RAW_VALUE]
                   [--scale FLOAT] [--edits] DNG_PATH ...
            Writes report.json and sRGB JPEG renders. Scale defaults to 0.125;
            use 1 for full resolution. Edits test exposure -1/+1 EV and WB +1000 K.
            An explicitly requested unsupported decoder is reported without fallback.
            Exit codes: 0 success, 1 render failure, 2 unsupported decoder, 64 usage error.
            """)
            exit(0)
        case "--edits":
            edits = true
        case "--output", "--decoder", "--scale":
            index += 1
            guard index < arguments.count else {
                throw ValidationError("Missing value after \(argument)")
            }
            let value = arguments[index]
            if argument == "--output" {
                output = URL(fileURLWithPath: value, isDirectory: true).standardizedFileURL
            } else if argument == "--decoder" {
                decoder = value
            } else {
                guard let parsed = Float(value), parsed.isFinite, parsed > 0, parsed <= 1 else {
                    throw ValidationError("--scale must be greater than 0 and at most 1")
                }
                scale = parsed
            }
        default:
            guard !argument.hasPrefix("--") else {
                throw ValidationError("Unknown option \(argument)")
            }
            inputs.append(URL(fileURLWithPath: argument).standardizedFileURL)
        }
        index += 1
    }
    guard let output, !inputs.isEmpty else {
        throw ValidationError("--output DIR and at least one DNG path are required; see --help")
    }
    return Options(output: output, decoder: decoder, scale: scale, edits: edits, inputs: inputs)
}

struct Dimensions: Codable {
    let width: Int
    let height: Int
}

struct RenderReport: Codable {
    let name: String
    let dimensions: Dimensions
    let exposureEV: Float
    let temperatureK: Float
    let tint: Float
    let actualCGImageCreated: Bool
    let meanLinearRGB: [Float]
    let meanRGBIsFinite: Bool
    let meanRGBDifferenceFromBaseline: Float?
    let jpeg: String
}

struct FileReport: Codable {
    let input: String
    var status = "failed"
    var supportedDecoders: [String] = []
    var selectedDecoder: String?
    var nativeDimensions: Dimensions?
    var renders: [RenderReport] = []
    var error: String?
}

struct Report: Codable {
    let operatingSystem: String
    let requestedDecoder: String?
    let scale: Float
    let edits: Bool
    let passed: Int
    let failed: Int
    let unsupported: Int
    let files: [FileReport]
}

func render(
    filter: CIRAWFilter, context: CIContext, name: String, destination: URL,
    baselineMean: [Float]?
) throws -> RenderReport {
    guard let image = filter.outputImage else {
        throw ValidationError("\(name): CIRAWFilter.outputImage is nil")
    }
    let extent = image.extent
    guard !extent.isInfinite, !extent.isNull,
        extent.width.isFinite, extent.height.isFinite,
        extent.width >= 1, extent.height >= 1
    else {
        throw ValidationError("\(name): invalid output extent \(extent)")
    }
    let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let cgImage = context.createCGImage(image, from: extent, format: .RGBA8, colorSpace: srgb)
    else {
        throw ValidationError("\(name): actual CGImage rendering failed")
    }

    // Summarize the floating-point render, before JPEG quantization. This
    // checks a finite RGB mean; it does not claim every pixel is finite.
    let average = image.applyingFilter(
        "CIAreaAverage", parameters: [kCIInputExtentKey: CIVector(cgRect: extent)])
    var mean = [Float](repeating: 0, count: 4)
    mean.withUnsafeMutableBytes { bytes in
        context.render(
            average, toBitmap: bytes.baseAddress!, rowBytes: 4 * MemoryLayout<Float>.size,
            bounds: CGRect(x: 0, y: 0, width: 1, height: 1), format: .RGBAf,
            colorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!)
    }
    guard mean.allSatisfy({ $0.isFinite }) else {
        throw ValidationError("\(name): non-finite floating-point mean")
    }
    let rgb = Array(mean.prefix(3))
    guard let writer = CGImageDestinationCreateWithURL(
        destination as CFURL, UTType.jpeg.identifier as CFString, 1, nil)
    else {
        throw ValidationError("Cannot create JPEG at \(destination.path)")
    }
    CGImageDestinationAddImage(
        writer, cgImage, [kCGImageDestinationLossyCompressionQuality: 0.93] as CFDictionary)
    guard CGImageDestinationFinalize(writer) else {
        throw ValidationError("Could not finish JPEG at \(destination.path)")
    }
    let difference = baselineMean.map { baseline in
        zip(rgb, baseline).reduce(Float(0)) { $0 + abs($1.0 - $1.1) } / 3
    }
    return RenderReport(
        name: name, dimensions: Dimensions(width: cgImage.width, height: cgImage.height),
        exposureEV: filter.exposure, temperatureK: filter.neutralTemperature,
        tint: filter.neutralTint, actualCGImageCreated: true, meanLinearRGB: rgb,
        meanRGBIsFinite: true, meanRGBDifferenceFromBaseline: difference, jpeg: destination.path)
}

func validate(_ input: URL, index: Int, options: Options, context: CIContext) -> FileReport {
    var result = FileReport(input: input.path)
    guard FileManager.default.fileExists(atPath: input.path) else {
        result.error = "Input file does not exist"
        return result
    }
    guard let filter = CIRAWFilter(imageURL: input, options: nil) else {
        result.error = "CIRAWFilter could not open input"
        return result
    }
    result.supportedDecoders = filter.supportedDecoderVersions.map(\.rawValue)
    let native = filter.nativeSize
    if native.width.isFinite, native.height.isFinite, native.width > 0, native.height > 0 {
        result.nativeDimensions = Dimensions(width: Int(native.width), height: Int(native.height))
    }
    if let requested = options.decoder {
        guard let supported = filter.supportedDecoderVersions.first(where: { $0.rawValue == requested })
        else {
            result.status = "unsupported_decoder"
            result.error = "Requested decoder \(requested) is not supported for this file; no fallback used"
            return result
        }
        filter.decoderVersion = supported
        guard filter.decoderVersion.rawValue == requested else {
            result.error = "Decoder selection did not retain requested version \(requested)"
            return result
        }
    }
    result.selectedDecoder = filter.decoderVersion.rawValue
    filter.scaleFactor = options.scale
    let exposure = filter.exposure
    let temperature = filter.neutralTemperature
    let tint = filter.neutralTint
    var variants: [(String, Float, Float)] = [("baseline", exposure, temperature)]
    if options.edits {
        variants += [
            ("exposure-minus-1", exposure - 1, temperature),
            ("exposure-plus-1", exposure + 1, temperature),
            ("white-balance-plus-1000K", exposure, temperature + 1000),
        ]
    }
    let stem = String(format: "%03d", index + 1) + "-" + input.deletingPathExtension().lastPathComponent
    do {
        for (name, ev, kelvin) in variants {
            filter.exposure = ev
            filter.neutralTemperature = kelvin
            filter.neutralTint = tint
            let destination = options.output.appendingPathComponent("\(stem)-\(name).jpg")
            result.renders.append(try render(
                filter: filter, context: context, name: name, destination: destination,
                baselineMean: result.renders.first?.meanLinearRGB))
        }
        result.status = "passed"
    } catch {
        result.error = String(describing: error)
    }
    return result
}

let options: Options
do {
    options = try parseOptions()
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(64)
}

do {
    try FileManager.default.createDirectory(at: options.output, withIntermediateDirectories: true)
    let context = CIContext(options: [.cacheIntermediates: false])
    var files: [FileReport] = []
    for (index, input) in options.inputs.enumerated() {
        let result = autoreleasepool {
            validate(input, index: index, options: options, context: context)
        }
        files.append(result)
        context.clearCaches()
        FileHandle.standardError.write(Data("[\(index + 1)/\(options.inputs.count)] \(result.status): \(input.lastPathComponent)\n".utf8))
    }
    let report = Report(
        operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
        requestedDecoder: options.decoder, scale: options.scale, edits: options.edits,
        passed: files.filter { $0.status == "passed" }.count,
        failed: files.filter { $0.status == "failed" }.count,
        unsupported: files.filter { $0.status == "unsupported_decoder" }.count,
        files: files)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let json = try encoder.encode(report)
    try json.write(to: options.output.appendingPathComponent("report.json"), options: .atomic)
    FileHandle.standardOutput.write(json)
    FileHandle.standardOutput.write(Data("\n".utf8))
    exit(report.failed > 0 ? 1 : report.unsupported > 0 ? 2 : 0)
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
