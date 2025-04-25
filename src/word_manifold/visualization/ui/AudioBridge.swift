import Foundation
import PythonKit

class AudioBridge {
    static let shared = AudioBridge()
    private let sys = Python.import("sys")
    private let os = Python.import("os")
    private let json = Python.import("json")
    private var audioGenerator: PythonObject?
    
    private init() {
        // Add Python module path
        let currentPath = FileManager.default.currentDirectoryPath
        sys.path.append(currentPath + "/src")
        
        do {
            // Import our Python module
            audioGenerator = Python.import("word_manifold.core.audio")
        } catch {
            print("Error importing Python module: \(error)")
        }
    }
    
    func generateAudio(parameters: [String: Any]) -> URL? {
        guard let audioGenerator = audioGenerator else { return nil }
        
        do {
            // Convert parameters to Python dict
            let jsonData = try JSONSerialization.data(withJSONObject: parameters)
            let jsonStr = String(data: jsonData, encoding: .utf8)!
            let pythonDict = json.loads(jsonStr)
            
            // Create AudioParams object
            let audioParams = audioGenerator.AudioParams(
                sample_rate: pythonDict["sample_rate"] as? Int ?? 44100,
                duration: pythonDict["duration"] as? Double ?? 5.0,
                base_freq: pythonDict["base_freq"] as? Double ?? 432.0,
                harmonics: pythonDict["harmonics"],
                ratios: pythonDict["ratios"],
                amplitude: pythonDict["amplitude"] as? Double ?? 0.5,
                fade_duration: pythonDict["fade_duration"] as? Double ?? 0.1,
                wave_shape: pythonDict["wave_shape"],
                modulation_type: pythonDict["modulation_type"],
                modulation_depth: pythonDict["modulation_depth"] as? Double ?? 0.2,
                modulation_freq: pythonDict["modulation_freq"] as? Double ?? 0.5,
                use_fibonacci: pythonDict["use_fibonacci"] as? Bool ?? true,
                use_phi: pythonDict["use_phi"] as? Bool ?? true,
                pulse_width: pythonDict["pulse_width"] as? Double ?? 0.5,
                resonance: pythonDict["resonance"] as? Double ?? 1.0,
                phase_shift: pythonDict["phase_shift"] as? Double ?? 0.0
            )
            
            // Generate audio
            let (_, audioData) = tuple(audioGenerator.generate_sacred_audio(audioParams))
            
            // Create temporary file
            let tempDir = FileManager.default.temporaryDirectory
            let fileName = "sacred_audio_\(Int(Date().timeIntervalSince1970)).wav"
            let fileURL = tempDir.appendingPathComponent(fileName)
            
            // Save audio
            audioGenerator.save_audio(audioData, audioParams.sample_rate, fileURL.path)
            
            return fileURL
        } catch {
            print("Error generating audio: \(error)")
            return nil
        }
    }
    
    func generateVariation(
        pattern: String,
        rhythmParams: RhythmParameters,
        recursiveParams: RecursiveParameters,
        delayParams: DelayParameters,
        harmonicParams: HarmonicParameters
    ) -> URL? {
        // Convert Swift parameters to Python dictionary
        let parameters: [String: Any] = [
            "pattern_type": pattern,
            "sample_rate": 44100,
            "duration": 5.0,
            "base_freq": harmonicParams.fundamental,
            "tempo": rhythmParams.tempo,
            "recursive": [
                "depth": recursiveParams.depth,
                "decay": recursiveParams.decay,
                "mutation_rate": recursiveParams.mutationRate,
                "self_similarity": recursiveParams.selfSimilarity,
                "evolution_rate": recursiveParams.evolutionRate
            ],
            "delay": [
                "delay_time": delayParams.delayTime,
                "feedback": delayParams.feedback,
                "mix": delayParams.mix,
                "filter_freq": delayParams.filterFreq,
                "resonance": delayParams.resonance
            ],
            "harmonic": [
                "fundamental": harmonicParams.fundamental,
                "phase_coherence": harmonicParams.phaseCoherence,
                "spectral_tilt": harmonicParams.spectralTilt,
                "selected_overtones": Array(harmonicParams.selectedOvertones),
                "selected_ratios": Array(harmonicParams.selectedRatios)
            ],
            "rhythm": [
                "tempo": rhythmParams.tempo,
                "subdivision": rhythmParams.subdivision,
                "pattern_length": rhythmParams.patternLength,
                "accent_probability": rhythmParams.accentProbability,
                "swing": rhythmParams.swing,
                "phase_shift": rhythmParams.phaseShift
            ]
        ]
        
        return generateAudio(parameters: parameters)
    }
}

// MARK: - Python Utilities
private func tuple(_ pythonTuple: PythonObject) -> (PythonObject, PythonObject) {
    return (pythonTuple[0], pythonTuple[1])
} 