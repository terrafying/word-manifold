import Foundation
import AVFoundation
import Combine

class AudioViewModel: ObservableObject {
    @Published var rhythmParams = RhythmParameters()
    @Published var recursiveParams = RecursiveParameters()
    @Published var delayParams = DelayParameters()
    @Published var harmonicParams = HarmonicParameters()
    @Published var isPlaying = false
    @Published var selectedPattern = "fibonacci"
    @Published var audioLevel: Float = 0.0
    @Published var isGenerating = false
    @Published var error: String?
    
    private var audioEngine: AVAudioEngine?
    private var audioPlayer: AVAudioPlayerNode?
    private var audioFile: AVAudioFile?
    private var displayLink: CADisplayLink?
    private var currentBuffer: AVAudioPCMBuffer?
    
    private let patterns = [
        "fibonacci", "golden", "euclidean", "spiral",
        "fib_golden", "spiral_euclidean", "harmonic_base",
        "harmonic_rhythm", "harmonic_delay", "modulated"
    ]
    
    init() {
        setupAudioEngine()
        setupDisplayLink()
    }
    
    private func setupAudioEngine() {
        audioEngine = AVAudioEngine()
        audioPlayer = AVAudioPlayerNode()
        
        guard let engine = audioEngine,
              let player = audioPlayer else { return }
        
        // Setup audio engine
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: nil)
        
        do {
            try engine.start()
        } catch {
            self.error = "Could not start audio engine: \(error.localizedDescription)"
        }
    }
    
    private func setupDisplayLink() {
        displayLink = CADisplayLink(target: self, selector: #selector(updateAudioLevel))
        displayLink?.add(to: .current, forMode: .common)
    }
    
    @objc private func updateAudioLevel() {
        guard let player = audioPlayer, player.isPlaying else {
            audioLevel = 0
            return
        }
        
        // Get current audio level
        let level = audioEngine?.mainMixerNode.outputVolume ?? 0
        audioLevel = level
    }
    
    func togglePlayback() {
        if isPlaying {
            stopPlayback()
        } else {
            generateAndPlay()
        }
    }
    
    private func stopPlayback() {
        audioPlayer?.stop()
        isPlaying = false
    }
    
    private func generateAndPlay() {
        isGenerating = true
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            // Generate audio using bridge
            if let audioURL = AudioBridge.shared.generateVariation(
                pattern: self.selectedPattern,
                rhythmParams: self.rhythmParams,
                recursiveParams: self.recursiveParams,
                delayParams: self.delayParams,
                harmonicParams: self.harmonicParams
            ) {
                do {
                    // Load audio file
                    let file = try AVAudioFile(forReading: audioURL)
                    let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                                frameCapacity: AVAudioFrameCount(file.length))
                    try file.read(into: buffer!)
                    
                    // Update on main thread
                    DispatchQueue.main.async {
                        self.currentBuffer = buffer
                        self.playBuffer()
                        self.isGenerating = false
                        self.isPlaying = true
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.error = "Error loading audio: \(error.localizedDescription)"
                        self.isGenerating = false
                    }
                }
            } else {
                DispatchQueue.main.async {
                    self.error = "Could not generate audio"
                    self.isGenerating = false
                }
            }
        }
    }
    
    private func playBuffer() {
        guard let player = audioPlayer,
              let buffer = currentBuffer else { return }
        
        player.stop()
        player.scheduleBuffer(buffer, at: nil, options: .loops)
        player.play()
    }
    
    func updateDelayParameters() {
        // Regenerate audio with new parameters if playing
        if isPlaying {
            generateAndPlay()
        }
    }
    
    deinit {
        displayLink?.invalidate()
        audioEngine?.stop()
    }
} 