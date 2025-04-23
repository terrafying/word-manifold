// Global state
const state = {
    isRotating: false,
    showTrails: false,
    showForceField: false,
    rotationSpeed: 50,
    plot: null,
    data: null,
    layout: null
};

// DOM Elements
const elements = {
    plot: document.getElementById('plot'),
    input: document.getElementById('input'),
    analyzeBtn: document.getElementById('analyze-btn'),
    rotateBtn: document.getElementById('rotate-btn'),
    trailsBtn: document.getElementById('trails-btn'),
    forceBtn: document.getElementById('force-btn'),
    resetBtn: document.getElementById('reset-btn'),
    speedSlider: document.getElementById('speed-slider'),
    status: document.getElementById('status'),
    dimensionality: document.getElementById('dimensionality'),
    dataPoints: document.getElementById('data-points'),
    avgDistance: document.getElementById('avg-distance'),
    neighborsList: document.getElementById('neighbors-list')
};

// Polarity elements
const polarityElements = {
    celestial: {
        score: document.getElementById('celestial-score'),
        value: document.getElementById('celestial-value')
    },
    vertical: {
        score: document.getElementById('vertical-score'),
        value: document.getElementById('vertical-value')
    },
    moral: {
        score: document.getElementById('moral-score'),
        value: document.getElementById('moral-value')
    },
    temporal: {
        score: document.getElementById('temporal-score'),
        value: document.getElementById('temporal-value')
    },
    energetic: {
        score: document.getElementById('energetic-score'),
        value: document.getElementById('energetic-value')
    }
};

// Initialize Plotly layout
const defaultLayout = {
    scene: {
        camera: {
            eye: { x: 1.5, y: 1.5, z: 1.5 }
        },
        xaxis: { title: 'X' },
        yaxis: { title: 'Y' },
        zaxis: { title: 'Z' }
    },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    showlegend: true,
    hovermode: 'closest'
};

// Helper Functions
function showStatus(message, type = 'info') {
    elements.status.textContent = message;
    elements.status.className = `status ${type}`;
    elements.status.style.display = 'block';
    
    setTimeout(() => {
        elements.status.style.display = 'none';
    }, 3000);
}

function updateMetrics(data) {
    elements.dimensionality.textContent = data.dimensions || '-';
    elements.dataPoints.textContent = data.points || '-';
    elements.avgDistance.textContent = data.avgDistance ? data.avgDistance.toFixed(2) : '-';
}

function updatePolarityMetrics(polarities) {
    for (const [key, value] of Object.entries(polarities)) {
        if (polarityElements[key]) {
            const score = (value * 100).toFixed(1);
            polarityElements[key].score.textContent = `${score}%`;
            polarityElements[key].value.style.width = `${Math.abs(score)}%`;
            polarityElements[key].value.style.backgroundColor = value > 0 ? '#4CAF50' : '#f44336';
        }
    }
}

function updateNeighbors(neighbors) {
    elements.neighborsList.innerHTML = '';
    neighbors.forEach(neighbor => {
        const li = document.createElement('li');
        li.textContent = `${neighbor.term} (${neighbor.distance.toFixed(3)})`;
        elements.neighborsList.appendChild(li);
    });
}

// Animation Functions
function startRotation() {
    if (!state.plot) return;
    
    const speed = state.rotationSpeed / 1000;
    let frame = 0;
    
    function rotate() {
        if (!state.isRotating) return;
        
        frame += speed;
        const layout = {
            ...state.layout,
            scene: {
                ...state.layout.scene,
                camera: {
                    eye: {
                        x: 1.5 * Math.cos(frame),
                        y: 1.5 * Math.sin(frame),
                        z: 1.5
                    }
                }
            }
        };
        
        Plotly.relayout(elements.plot, layout);
        requestAnimationFrame(rotate);
    }
    
    rotate();
}

// Event Handlers
elements.analyzeBtn.addEventListener('click', async () => {
    const text = elements.input.value.trim();
    if (!text) {
        showStatus('Please enter text to analyze', 'error');
        return;
    }
    
    try {
        showStatus('Analyzing text...');
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        
        if (!response.ok) throw new Error('Analysis failed');
        
        const data = await response.json();
        state.data = data;
        
        // Update visualization
        await updateVisualization(data);
        showStatus('Analysis complete', 'success');
        
    } catch (error) {
        console.error('Analysis error:', error);
        showStatus('Failed to analyze text', 'error');
    }
});

elements.rotateBtn.addEventListener('click', () => {
    state.isRotating = !state.isRotating;
    elements.rotateBtn.classList.toggle('active');
    
    if (state.isRotating) {
        startRotation();
        showStatus('Auto-rotation enabled');
    } else {
        showStatus('Auto-rotation disabled');
    }
});

elements.trailsBtn.addEventListener('click', () => {
    state.showTrails = !state.showTrails;
    elements.trailsBtn.classList.toggle('active');
    
    if (state.data) {
        updateVisualization(state.data);
    }
    
    showStatus(`Trails ${state.showTrails ? 'enabled' : 'disabled'}`);
});

elements.forceBtn.addEventListener('click', () => {
    state.showForceField = !state.showForceField;
    elements.forceBtn.classList.toggle('active');
    
    if (state.data) {
        updateVisualization(state.data);
    }
    
    showStatus(`Force field ${state.showForceField ? 'enabled' : 'disabled'}`);
});

elements.resetBtn.addEventListener('click', () => {
    if (state.plot) {
        Plotly.relayout(elements.plot, defaultLayout);
        showStatus('View reset to default');
    }
});

elements.speedSlider.addEventListener('input', (event) => {
    state.rotationSpeed = parseInt(event.target.value);
});

// Visualization Functions
async function updateVisualization(data) {
    const traces = [];
    
    // Main data points
    traces.push({
        type: 'scatter3d',
        mode: 'markers+text',
        x: data.coordinates.map(p => p[0]),
        y: data.coordinates.map(p => p[1]),
        z: data.coordinates.map(p => p[2]),
        text: data.labels,
        textposition: 'top center',
        marker: {
            size: 6,
            color: data.colors || '#1f77b4',
            opacity: 0.8
        },
        name: 'Words'
    });
    
    // Add trails if enabled
    if (state.showTrails) {
        traces.push({
            type: 'scatter3d',
            mode: 'lines',
            x: data.trails.map(p => p[0]),
            y: data.trails.map(p => p[1]),
            z: data.trails.map(p => p[2]),
            line: {
                color: '#888',
                width: 1,
                dash: 'dot'
            },
            opacity: 0.3,
            name: 'Trails'
        });
    }
    
    // Add force field if enabled
    if (state.showForceField) {
        traces.push({
            type: 'cone',
            x: data.forces.map(f => f.x),
            y: data.forces.map(f => f.y),
            z: data.forces.map(f => f.z),
            u: data.forces.map(f => f.u),
            v: data.forces.map(f => f.v),
            w: data.forces.map(f => f.w),
            colorscale: 'Viridis',
            name: 'Force Field'
        });
    }
    
    // Update or create plot
    if (state.plot) {
        await Plotly.react(elements.plot, traces, state.layout);
    } else {
        state.layout = { ...defaultLayout };
        state.plot = await Plotly.newPlot(elements.plot, traces, state.layout);
    }
    
    // Update metrics
    updateMetrics(data.metrics);
    updatePolarityMetrics(data.polarities);
    updateNeighbors(data.neighbors);
}

// Initialize the visualization
document.addEventListener('DOMContentLoaded', () => {
    showStatus('Ready to analyze text');
}); 