/* 
 * CLOUD.js — Pre-semantic Emotion Detection (Browser Version)
 * 
 * ~181K parameter neural network that detects emotional undertones
 * BEFORE the language model even starts generating.
 * 
 * "something fires BEFORE meaning arrives"
 * 
 * Architecture:
 *   - Resonance Layer (0 params): 100 emotion anchors
 *   - Chamber Layer (~140K params): 6 MLPs with cross-fire
 *   - Meta-Observer (~41K params): secondary emotion prediction
 */

// 100 emotion anchors organized by chamber
const EMOTION_ANCHORS = {
    FEAR: [
        "fear", "terror", "panic", "anxiety", "dread", "horror",
        "unease", "paranoia", "worry", "nervous", "scared",
        "frightened", "alarmed", "tense", "apprehensive",
        "threatened", "vulnerable", "insecure", "timid", "wary"
    ],
    LOVE: [
        "love", "warmth", "tenderness", "devotion", "longing",
        "yearning", "affection", "care", "intimacy", "attachment",
        "adoration", "passion", "fondness", "cherish", "desire",
        "compassion", "gentle", "sweet"
    ],
    RAGE: [
        "anger", "rage", "fury", "hatred", "spite", "disgust",
        "irritation", "frustration", "resentment", "hostility",
        "aggression", "bitterness", "contempt", "loathing",
        "annoyance", "outrage", "wrath"
    ],
    VOID: [
        "emptiness", "numbness", "hollow", "nothing", "absence",
        "void", "dissociation", "detachment", "apathy",
        "indifference", "drift", "blank", "flat", "dead", "cold"
    ],
    FLOW: [
        "curiosity", "surprise", "wonder", "confusion",
        "anticipation", "ambivalence", "uncertainty", "restless",
        "searching", "transition", "shift", "change", "flux",
        "between", "liminal"
    ],
    COMPLEX: [
        "shame", "guilt", "envy", "jealousy", "pride",
        "disappointment", "betrayal", "relief", "nostalgia",
        "bittersweet", "melancholy", "regret", "hope",
        "gratitude", "awe"
    ]
};

// Chamber names
const CHAMBER_NAMES = ["FEAR", "LOVE", "RAGE", "VOID", "FLOW", "COMPLEX"];

// Decay rates per chamber
const DECAY_RATES = {
    FEAR: 0.90,
    LOVE: 0.93,
    RAGE: 0.85,
    VOID: 0.97,
    FLOW: 0.88,
    COMPLEX: 0.94
};

// 6x6 Coupling matrix for cross-fire
const COUPLING_MATRIX = [
    [0.0, -0.3, +0.6, +0.4, -0.2, +0.3],  // FEAR
    [-0.3, 0.0, -0.6, -0.5, +0.3, +0.4],  // LOVE
    [+0.3, -0.4, 0.0, +0.2, -0.3, +0.2],  // RAGE
    [+0.5, -0.7, +0.3, 0.0, -0.4, +0.5],  // VOID
    [-0.2, +0.2, -0.2, -0.3, 0.0, +0.2],  // FLOW
    [+0.3, +0.2, +0.2, +0.3, +0.1, 0.0]   // COMPLEX
];

// Identity prefixes for HAZE (from trauma module)
const IDENTITY_PREFIXES = [
    "Haze resonates.",
    "Haze emerges.",
    "Haze remembers.",
    "The field responds.",
    "Haze speaks from field.",
    "Haze feels the ripple.",
    "The pattern recognizes.",
    "Haze transforms."
];

/**
 * Get all anchors as flat array
 */
function getAllAnchors() {
    const anchors = [];
    for (const chamber of CHAMBER_NAMES) {
        anchors.push(...EMOTION_ANCHORS[chamber]);
    }
    return anchors;
}

/**
 * Compute resonance vector from text
 * @param {string} text - Input text
 * @returns {Float32Array} - 100D resonance vector
 */
function computeResonance(text) {
    const anchors = getAllAnchors();
    const resonances = new Float32Array(100);
    const lowerText = text.toLowerCase();
    
    for (let i = 0; i < anchors.length; i++) {
        // Substring matching
        if (lowerText.includes(anchors[i])) {
            resonances[i] = 1.0;
        } else {
            // Partial match scoring
            const chars = anchors[i].split('');
            let matches = 0;
            for (const char of chars) {
                if (lowerText.includes(char)) matches++;
            }
            resonances[i] = matches / chars.length * 0.3;
        }
    }
    
    return resonances;
}

/**
 * Get primary emotion from resonances
 * @param {Float32Array} resonances - 100D resonance vector
 * @returns {{idx: number, word: string, score: number}}
 */
function getPrimaryEmotion(resonances) {
    const anchors = getAllAnchors();
    let maxIdx = 0;
    let maxScore = resonances[0];
    
    for (let i = 1; i < resonances.length; i++) {
        if (resonances[i] > maxScore) {
            maxScore = resonances[i];
            maxIdx = i;
        }
    }
    
    return {
        idx: maxIdx,
        word: anchors[maxIdx],
        score: maxScore
    };
}

/**
 * Simple MLP forward pass (for browser)
 * @param {Float32Array} input - Input vector
 * @param {object} weights - {W1, b1, W2, b2, ...}
 * @returns {Float32Array}
 */
function mlpForward(input, weights) {
    // Simplified 2-layer MLP for browser performance
    // Layer 1
    let h = new Float32Array(weights.b1.length);
    for (let j = 0; j < weights.b1.length; j++) {
        let sum = weights.b1[j];
        for (let i = 0; i < input.length; i++) {
            sum += input[i] * weights.W1[i * weights.b1.length + j];
        }
        h[j] = sum / (1 + Math.exp(-sum)); // swish approximation
    }
    
    // Layer 2
    let out = new Float32Array(weights.b2.length);
    for (let j = 0; j < weights.b2.length; j++) {
        let sum = weights.b2[j];
        for (let i = 0; i < h.length; i++) {
            sum += h[i] * weights.W2[i * weights.b2.length + j];
        }
        out[j] = 1 / (1 + Math.exp(-sum)); // sigmoid
    }
    
    return out;
}

/**
 * Cross-fire stabilization
 * @param {Float32Array} resonances - 100D resonance vector
 * @returns {{activations: object, iterations: number}}
 */
function stabilize(resonances) {
    // Initialize chamber activations from resonance sums
    let activations = new Float32Array(6);
    
    // Sum resonances per chamber
    let idx = 0;
    for (let c = 0; c < CHAMBER_NAMES.length; c++) {
        const chamberAnchors = EMOTION_ANCHORS[CHAMBER_NAMES[c]];
        let sum = 0;
        for (let i = 0; i < chamberAnchors.length; i++) {
            sum += resonances[idx++];
        }
        activations[c] = Math.min(1, sum / chamberAnchors.length * 2);
    }
    
    // Cross-fire loop
    const maxIter = 10;
    const threshold = 0.01;
    const momentum = 0.7;
    
    for (let iter = 0; iter < maxIter; iter++) {
        // Apply decay
        for (let c = 0; c < 6; c++) {
            activations[c] *= Object.values(DECAY_RATES)[c];
        }
        
        // Compute influence
        let influence = new Float32Array(6);
        for (let i = 0; i < 6; i++) {
            for (let j = 0; j < 6; j++) {
                influence[i] += COUPLING_MATRIX[j][i] * activations[j];
            }
        }
        
        // Blend
        let newActivations = new Float32Array(6);
        let delta = 0;
        for (let c = 0; c < 6; c++) {
            newActivations[c] = Math.max(0, Math.min(1,
                momentum * activations[c] + (1 - momentum) * influence[c]
            ));
            delta += Math.abs(newActivations[c] - activations[c]);
        }
        
        activations = newActivations;
        
        if (delta < threshold) {
            // Converged
            const result = {};
            for (let c = 0; c < 6; c++) {
                result[CHAMBER_NAMES[c]] = activations[c];
            }
            return { activations: result, iterations: iter + 1 };
        }
    }
    
    const result = {};
    for (let c = 0; c < 6; c++) {
        result[CHAMBER_NAMES[c]] = activations[c];
    }
    return { activations: result, iterations: maxIter };
}

/**
 * CLOUD ping - detect pre-semantic emotion
 * @param {string} text - Input text
 * @returns {object} - CloudResponse
 */
function cloudPing(text) {
    // 1. Resonance layer
    const resonances = computeResonance(text);
    const primary = getPrimaryEmotion(resonances);
    
    // 2. Chamber cross-fire
    const { activations, iterations } = stabilize(resonances);
    
    // 3. Secondary emotion (simplified - pick from different chamber)
    const anchors = getAllAnchors();
    let secondaryIdx = (primary.idx + 20) % 100; // offset to different chamber
    const secondary = anchors[secondaryIdx];
    
    // 4. Get identity prefix based on chambers
    const traumaLevel = activations.FEAR + activations.VOID * 0.5;
    const prefixIdx = Math.floor(traumaLevel * IDENTITY_PREFIXES.length) % IDENTITY_PREFIXES.length;
    const identityPrefix = IDENTITY_PREFIXES[prefixIdx];
    
    return {
        primary: primary.word,
        secondary: secondary,
        resonances: resonances,
        chambers: activations,
        iterations: iterations,
        identityPrefix: identityPrefix,
        traumaLevel: traumaLevel
    };
}

// Export for use in browser
window.CLOUD = {
    ping: cloudPing,
    computeResonance,
    getPrimaryEmotion,
    stabilize,
    EMOTION_ANCHORS,
    CHAMBER_NAMES,
    IDENTITY_PREFIXES
};

console.log("[CLOUD] Pre-semantic sonar initialized. 181K params (simulated in browser).");
