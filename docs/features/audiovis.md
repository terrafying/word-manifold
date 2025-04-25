# Audio-Reactive Visualization

Word Manifold includes a powerful audio-reactive visualization system that creates mesmerizing ASCII art patterns that respond to audio input in real-time.

## Quick Start

```bash
# Start basic visualization
word-manifold audiovis

# Use high-resolution mode
word-manifold audiovis --high-res

# Specify audio device
word-manifold audiovis --device 1
```

## Features

### Audio Input

The visualization supports two types of audio input:

1. **Microphone Input**
   - Real-time audio capture from any system audio device
   - Configurable sample rate and buffer size
   - Automatic device selection

2. **Audio File Playback**
   - Support for common audio formats (WAV, MP3, etc.)
   - Waveform visualization
   - Playback controls (play, pause, seek)

### Visualization Patterns

#### Wave Pattern
```bash
word-manifold audiovis --pattern wave
```
- Classic waveform visualization
- Frequency-based modulation
- Optional interference patterns
- Adjustable complexity and density

#### Mandala Pattern
```bash
word-manifold audiovis --pattern mandala
```
- Circular patterns that respond to audio features
- Multiple layers with independent animation
- Beat-reactive scaling and rotation
- Configurable symmetry and complexity

#### Field Pattern
```bash
word-manifold audiovis --pattern field
```
- Dynamic noise field visualization
- Audio-reactive thresholding
- Complex wave interactions
- Density and complexity controls

### High-Resolution Mode

Enable detailed ASCII patterns with more characters:

```bash
word-manifold audiovis --high-res
```

High-res mode features:
- Extended ASCII character set
- More detailed patterns
- Gradient effects
- Better visual quality

### Visual Effects

1. **Mirror Effect**
   ```bash
   word-manifold audiovis --effects mirror
   ```
   - Mirrors the pattern horizontally
   - Creates symmetrical designs

2. **Pulse Effect**
   ```bash
   word-manifold audiovis --effects pulse
   ```
   - Responds to detected beats
   - Creates visual emphasis

3. **Glow Effect**
   ```bash
   word-manifold audiovis --effects glow
   ```
   - Adds text shadow animation
   - Creates depth perception

### Color Modes

1. **Spectrum Mode**
   - Colors based on frequency content
   - Smooth transitions across spectrum

2. **Intensity Mode**
   - Brightness based on audio intensity
   - Dynamic range visualization

3. **Rainbow Mode**
   - Cycling color patterns
   - Phase-based transitions

4. **Custom Colors**
   - User-defined color schemes
   - RGB and HSL support

## Audio Analysis

The visualization performs several types of audio analysis:

1. **Spectral Analysis**
   - Real-time FFT computation
   - Frequency band extraction
   - Spectral flux measurement

2. **Beat Detection**
   - Energy-based beat detection
   - Onset detection
   - Tempo estimation

3. **Feature Extraction**
   - Intensity measurement
   - Dominant frequency tracking
   - Spectral centroid

## Interactive Controls

The web interface provides various controls:

1. **Pattern Controls**
   - Pattern style selection
   - Complexity adjustment
   - Density settings
   - Speed control

2. **Audio Controls**
   - Input source selection
   - Volume control
   - Playback controls
   - Device selection

3. **Effect Controls**
   - Effect toggles
   - Effect parameters
   - Color mode selection
   - Custom color picker

4. **Visualization Settings**
   - Size adjustment
   - Resolution control
   - Performance settings
   - Display options

## Advanced Usage

### Custom Templates

You can use custom HTML templates for visualization:

```bash
word-manifold audiovis --template my_template.html
```

### WebSocket API

The visualization server provides a WebSocket API for real-time data:

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // data.pattern: ASCII pattern
    // data.intensity: Audio intensity
    // data.dominant_freq: Dominant frequency
    // data.spectrum: Frequency spectrum
    // data.beat_detected: Beat detection state
};
```

### Configuration

Create a configuration file for persistent settings:

```yaml
# audiovis_config.yml
audio:
  sample_rate: 44100
  block_size: 2048
  device: 1

visualization:
  pattern: mandala
  high_res: true
  effects:
    - mirror
    - glow
  color_mode: spectrum
```

```bash
word-manifold audiovis --config audiovis_config.yml
```

## Performance Tips

1. **Adjust Buffer Size**
   ```bash
   word-manifold audiovis --block-size 1024
   ```
   - Smaller values: Lower latency but higher CPU usage
   - Larger values: Higher latency but lower CPU usage

2. **Frame Rate Control**
   ```bash
   word-manifold audiovis --fps 30
   ```
   - Balance between smoothness and performance

3. **Pattern Complexity**
   ```bash
   word-manifold audiovis --complexity 5
   ```
   - Lower values for better performance
   - Higher values for more detail

4. **Resolution Settings**
   ```bash
   word-manifold audiovis --width 80 --height 40
   ```
   - Adjust pattern size for performance

## Troubleshooting

### Common Issues

1. **Audio Device Not Found**
   ```bash
   # List available audio devices
   word-manifold audiovis --list-devices
   
   # Specify device explicitly
   word-manifold audiovis --device 1
   ```

2. **High CPU Usage**
   - Reduce pattern complexity
   - Lower the frame rate
   - Disable high-res mode
   - Use smaller pattern size

3. **WebSocket Connection Issues**
   - Check port availability
   - Verify network settings
   - Use different port:
     ```bash
     word-manifold audiovis --port 8766
     ```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
word-manifold audiovis --debug
```

## Examples

### Basic Visualization
```bash
# Start with default settings
word-manifold audiovis
```

### Complex Pattern
```bash
# High-res mandala with effects
word-manifold audiovis \
    --pattern mandala \
    --high-res \
    --complexity 8 \
    --effects mirror pulse \
    --color-mode rainbow
```

### Audio File Analysis
```bash
# Visualize audio file with custom settings
word-manifold audiovis \
    --pattern field \
    --audio music.mp3 \
    --density 75 \
    --effects glow
```

### Custom Setup
```bash
# Full customization
word-manifold audiovis \
    --pattern wave \
    --high-res \
    --device 1 \
    --sample-rate 48000 \
    --block-size 1024 \
    --fps 60 \
    --width 120 \
    --height 60 \
    --effects mirror pulse glow \
    --color-mode spectrum
``` 