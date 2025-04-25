import SwiftUI
import AVFoundation
import AudioKit
import AudioToolbox

// MARK: - Data Models
struct RhythmParameters: Codable {
    var tempo: Double = 120.0
    var subdivision: Int = 4
    var patternLength: Int = 16
    var accentProbability: Double = 0.3
    var swing: Double = 0.0
    var phaseShift: Double = 0.0
}

struct RecursiveParameters: Codable {
    var depth: Int = 3
    var decay: Double = 0.7
    var mutationRate: Double = 0.2
    var selfSimilarity: Double = 0.8
    var evolutionRate: Double = 0.3
}

struct DelayParameters: Codable {
    var delayTime: Double = 0.25
    var feedback: Double = 0.4
    var mix: Double = 0.3
    var filterFreq: Double = 2000.0
    var resonance: Double = 0.7
}

struct HarmonicParameters: Codable {
    var fundamental: Double = 432.0
    var phaseCoherence: Double = 0.8
    var spectralTilt: Double = -6.0
    var selectedOvertones: Set<Int> = [1, 2, 3, 5, 8]
    var selectedRatios: Set<Int> = [1, 2, 3, 5]
}

// MARK: - View Models
class AudioGeneratorViewModel: ObservableObject {
    @Published var rhythmParams = RhythmParameters()
    @Published var recursiveParams = RecursiveParameters()
    @Published var delayParams = DelayParameters()
    @Published var harmonicParams = HarmonicParameters()
    @Published var isPlaying = false
    @Published var selectedPattern = "fibonacci"
    @Published var audioLevel: Float = 0.0
    
    private var audioEngine: AVAudioEngine?
    private var patternNode: AVAudioSourceNode?
    private var delayNode: AVAudioUnitDelay?
    private var filterNode: AVAudioUnitEQ?
    
    let patterns = [
        "fibonacci", "golden", "euclidean", "spiral",
        "fib_golden", "spiral_euclidean", "harmonic_base",
        "harmonic_rhythm", "harmonic_delay", "modulated"
    ]
    
    func setupAudio() {
        audioEngine = AVAudioEngine()
        guard let engine = audioEngine else { return }
        
        // Create nodes
        let mainMixer = engine.mainMixerNode
        let output = engine.outputNode
        let format = output.inputFormat(forBus: 0)
        
        // Pattern generator node
        patternNode = AVAudioSourceNode { _, _, frameCount, audioBufferList in
            let ablPointer = UnsafeMutableAudioBufferListPointer(audioBufferList)
            let frameLength = Int(frameCount)
            
            // Generate audio data based on current parameters
            for frame in 0..<frameLength {
                let sample = self.generateSample(at: Double(frame) / format.sampleRate)
                for buffer in ablPointer {
                    let buf = UnsafeMutableBufferPointer<Float>(buffer)
                    buf[frame] = Float(sample)
                }
            }
            return noErr
        }
        
        // Setup delay
        delayNode = AVAudioUnitDelay()
        delayNode?.delayTime = delayParams.delayTime
        delayNode?.feedback = Float(delayParams.feedback)
        delayNode?.wetDryMix = Float(delayParams.mix * 100)
        
        // Setup filter
        filterNode = AVAudioUnitEQ(numberOfBands: 1)
        if let filter = filterNode?.bands[0] {
            filter.filterType = .lowPass
            filter.frequency = Float(delayParams.filterFreq)
            filter.bandwidth = Float(delayParams.resonance)
        }
        
        // Connect nodes
        if let pattern = patternNode {
            engine.attach(pattern)
            engine.connect(pattern, to: delayNode!, format: format)
            engine.connect(delayNode!, to: filterNode!, format: format)
            engine.connect(filterNode!, to: mainMixer, format: format)
            engine.connect(mainMixer, to: output, format: format)
        }
        
        // Start engine
        do {
            try engine.start()
        } catch {
            print("Could not start engine: \(error.localizedDescription)")
        }
    }
    
    func generateSample(at time: Double) -> Double {
        // This is where we'll generate the actual audio sample
        // based on the current parameters and selected pattern
        var sample = 0.0
        
        // Basic oscillator for testing
        let frequency = harmonicParams.fundamental
        sample = sin(2.0 * .pi * frequency * time)
        
        return sample
    }
    
    func togglePlayback() {
        isPlaying.toggle()
        if isPlaying {
            setupAudio()
        } else {
            audioEngine?.stop()
        }
    }
    
    func updateDelayParameters() {
        delayNode?.delayTime = delayParams.delayTime
        delayNode?.feedback = Float(delayParams.feedback)
        delayNode?.wetDryMix = Float(delayParams.mix * 100)
        
        if let filter = filterNode?.bands[0] {
            filter.frequency = Float(delayParams.filterFreq)
            filter.bandwidth = Float(delayParams.resonance)
        }
    }
}

// MARK: - Views
struct WaveformView: View {
    @Binding var audioLevel: Float
    
    var body: some View {
        GeometryReader { geometry in
            Path { path in
                let width = geometry.size.width
                let height = geometry.size.height
                let midY = height / 2
                
                path.move(to: CGPoint(x: 0, y: midY))
                
                for x in 0..<Int(width) {
                    let progress = Double(x) / Double(width)
                    let y = midY + CGFloat(sin(progress * 2 * .pi) * Double(audioLevel) * height / 2)
                    path.addLine(to: CGPoint(x: CGFloat(x), y: y))
                }
            }
            .stroke(Color.blue, lineWidth: 2)
        }
    }
}

struct ParameterSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    
    var body: some View {
        VStack(alignment: .leading) {
            Text(title)
                .font(.caption)
            HStack {
                Slider(value: $value, in: range, step: step)
                Text(String(format: "%.2f", value))
                    .font(.caption)
                    .frame(width: 50)
            }
        }
        .padding(.horizontal)
    }
}

struct PatternControlView: View {
    @ObservedObject var viewModel: AudioGeneratorViewModel
    
    var body: some View {
        VStack {
            Picker("Pattern", selection: $viewModel.selectedPattern) {
                ForEach(viewModel.patterns, id: \.self) { pattern in
                    Text(pattern).tag(pattern)
                }
            }
            .pickerStyle(MenuPickerStyle())
            .padding()
            
            WaveformView(audioLevel: $viewModel.audioLevel)
                .frame(height: 100)
                .padding()
            
            Button(action: viewModel.togglePlayback) {
                Image(systemName: viewModel.isPlaying ? "stop.circle.fill" : "play.circle.fill")
                    .font(.system(size: 44))
            }
            .padding()
        }
    }
}

struct RhythmControlView: View {
    @ObservedObject var viewModel: AudioGeneratorViewModel
    
    var body: some View {
        VStack {
            Text("Rhythm Parameters")
                .font(.headline)
            
            ParameterSlider(
                title: "Tempo",
                value: $viewModel.rhythmParams.tempo,
                range: 60...240,
                step: 1
            )
            
            ParameterSlider(
                title: "Swing",
                value: $viewModel.rhythmParams.swing,
                range: 0...0.33,
                step: 0.01
            )
            
            ParameterSlider(
                title: "Accent Probability",
                value: $viewModel.rhythmParams.accentProbability,
                range: 0...1,
                step: 0.05
            )
        }
        .padding()
    }
}

struct DelayControlView: View {
    @ObservedObject var viewModel: AudioGeneratorViewModel
    
    var body: some View {
        VStack {
            Text("Delay Parameters")
                .font(.headline)
            
            ParameterSlider(
                title: "Delay Time",
                value: $viewModel.delayParams.delayTime,
                range: 0...1,
                step: 0.01
            )
            
            ParameterSlider(
                title: "Feedback",
                value: $viewModel.delayParams.feedback,
                range: 0...0.95,
                step: 0.05
            )
            
            ParameterSlider(
                title: "Mix",
                value: $viewModel.delayParams.mix,
                range: 0...1,
                step: 0.05
            )
            
            ParameterSlider(
                title: "Filter Frequency",
                value: $viewModel.delayParams.filterFreq,
                range: 20...20000,
                step: 100
            )
        }
        .padding()
        .onChange(of: viewModel.delayParams) { _ in
            viewModel.updateDelayParameters()
        }
    }
}

struct HarmonicControlView: View {
    @ObservedObject var viewModel: AudioGeneratorViewModel
    
    var body: some View {
        VStack {
            Text("Harmonic Parameters")
                .font(.headline)
            
            ParameterSlider(
                title: "Fundamental",
                value: $viewModel.harmonicParams.fundamental,
                range: 20...1000,
                step: 1
            )
            
            ParameterSlider(
                title: "Phase Coherence",
                value: $viewModel.harmonicParams.phaseCoherence,
                range: 0...1,
                step: 0.05
            )
            
            ParameterSlider(
                title: "Spectral Tilt",
                value: $viewModel.harmonicParams.spectralTilt,
                range: -12...0,
                step: 0.5
            )
        }
        .padding()
    }
}

struct MainView: View {
    @StateObject var viewModel = AudioGeneratorViewModel()
    
    var body: some View {
        NavigationView {
            VStack {
                PatternControlView(viewModel: viewModel)
                
                ScrollView {
                    RhythmControlView(viewModel: viewModel)
                    DelayControlView(viewModel: viewModel)
                    HarmonicControlView(viewModel: viewModel)
                }
            }
            .navigationTitle("Sacred Audio Explorer")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

// MARK: - Preview
struct MainView_Previews: PreviewProvider {
    static var previews: some View {
        MainView()
    }
} 