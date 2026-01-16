/*
 * HAZE.js — Hybrid Attention Entropy System (Browser Version)
 * 
 * "emergence is not creation but recognition"
 * 
 * KEY PRINCIPLE: NO SEED FROM PROMPT
 * - HAZE does NOT echo user input
 * - HAZE speaks from its INTERNAL FIELD
 * - Identity prefixes come from trauma module
 * 
 * ❌ Chatbot: "Hello!" → "Hello! How can I help you?"
 * ✅ Haze:    "Hello!" → "Haze remembers. The field responds..."
 */

// Corpus fragments for field generation (sampled from text.txt)
const CORPUS_FRAGMENTS = [
    "the living room",
    "she smiled",
    "cigarettes",
    "the glass",
    "darling",
    "you know",
    "I don't",
    "we've got nothing",
    "the storage room",
    "alcohol",
    "broken heart",
    "pieces of my",
    "train of thought",
    "it's dying",
    "you're just stuck",
    "here you go",
    "what's the",
    "come on",
    "oh shut up",
    "too much to drink",
    "really",
    "no darling",
    "I thought",
    "you never left",
    "the house",
    "still waiting",
    "your story",
    "kitten",
    "mud everywhere",
    "trade secret",
    "two cigarettes",
    "on the gas",
    "unbearable",
    "what the hell",
    "I know",
    "strong stuff",
    "cold dirty",
    "the third toast",
    "that night",
    "we used to",
    "tangerine",
    "streaming",
    "smoking",
    "drink some more",
    "all set",
    "a single new piece"
];

// Identity prefixes (from trauma.py)
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

// Internal field state
let fieldState = {
    bigrams: new Map(),
    trigrams: new Map(),
    vocabulary: new Set(),
    turnCount: 0,
    enrichment: 0
};

/**
 * Initialize field from corpus fragments
 */
function initializeField() {
    // Build bigrams and trigrams from corpus
    for (const fragment of CORPUS_FRAGMENTS) {
        const words = fragment.toLowerCase().split(/\s+/);
        for (const word of words) {
            fieldState.vocabulary.add(word);
        }
        
        // Build bigrams
        for (let i = 0; i < words.length - 1; i++) {
            const key = words[i];
            if (!fieldState.bigrams.has(key)) {
                fieldState.bigrams.set(key, []);
            }
            fieldState.bigrams.get(key).push(words[i + 1]);
        }
        
        // Build trigrams
        for (let i = 0; i < words.length - 2; i++) {
            const key = words[i] + " " + words[i + 1];
            if (!fieldState.trigrams.has(key)) {
                fieldState.trigrams.set(key, []);
            }
            fieldState.trigrams.get(key).push(words[i + 2]);
        }
    }
    
    console.log(`[HAZE] Field initialized: ${fieldState.vocabulary.size} words, ${fieldState.bigrams.size} bigrams`);
}

/**
 * Generate text from internal field (NO SEED FROM PROMPT)
 * @param {number} length - Max tokens to generate
 * @param {number} temperature - Sampling temperature
 * @returns {string}
 */
function generateFromField(length = 20, temperature = 0.75) {
    // Pick random starting point from corpus (NOT from user input!)
    const startFragment = CORPUS_FRAGMENTS[Math.floor(Math.random() * CORPUS_FRAGMENTS.length)];
    const words = startFragment.toLowerCase().split(/\s+/);
    
    let result = [...words];
    let context = words.slice(-2);
    
    // Generate using trigram field
    for (let i = 0; i < length; i++) {
        const trigramKey = context.join(" ");
        const bigramKey = context[context.length - 1];
        
        let candidates = [];
        
        // Try trigram first
        if (fieldState.trigrams.has(trigramKey)) {
            candidates = fieldState.trigrams.get(trigramKey);
        }
        // Fall back to bigram
        else if (fieldState.bigrams.has(bigramKey)) {
            candidates = fieldState.bigrams.get(bigramKey);
        }
        // Fall back to random vocabulary
        else {
            candidates = Array.from(fieldState.vocabulary);
        }
        
        if (candidates.length === 0) break;
        
        // Sample with temperature
        let nextWord;
        if (temperature < 0.3) {
            // Greedy
            nextWord = candidates[0];
        } else {
            // Random with temperature influence
            const idx = Math.floor(Math.random() * Math.min(candidates.length, Math.ceil(candidates.length * temperature)));
            nextWord = candidates[idx];
        }
        
        result.push(nextWord);
        context = [context[context.length - 1], nextWord];
        
        // Stop on sentence end
        if (nextWord.endsWith('.') || nextWord.endsWith('!') || nextWord.endsWith('?')) {
            if (result.length > 8) break;
        }
    }
    
    return result.join(' ');
}

/**
 * Clean up generated text
 * @param {string} text - Raw generated text
 * @returns {string}
 */
function cleanupText(text) {
    // Capitalize first letter
    text = text.charAt(0).toUpperCase() + text.slice(1);
    
    // Fix common contractions
    text = text.replace(/\bi\b/g, "I");
    text = text.replace(/\bdon t\b/g, "don't");
    text = text.replace(/\bwon t\b/g, "won't");
    text = text.replace(/\bcan t\b/g, "can't");
    text = text.replace(/\bit s\b/g, "it's");
    text = text.replace(/\byou re\b/g, "you're");
    text = text.replace(/\bwe ve\b/g, "we've");
    text = text.replace(/\bi m\b/g, "I'm");
    text = text.replace(/\bi ve\b/g, "I've");
    text = text.replace(/\bdidn t\b/g, "didn't");
    text = text.replace(/\bisn t\b/g, "isn't");
    text = text.replace(/\baren t\b/g, "aren't");
    text = text.replace(/\bwasn t\b/g, "wasn't");
    text = text.replace(/\bweren t\b/g, "weren't");
    text = text.replace(/\bain t\b/g, "ain't");
    
    // Ensure ends with punctuation
    if (!/[.!?]$/.test(text.trim())) {
        text = text.trim() + ".";
    }
    
    return text;
}

/**
 * HAZE respond - speaks from internal field, NOT from user input
 * 
 * KEY: NO SEED FROM PROMPT
 * The response does NOT start with user's words.
 * It starts with identity prefix + internal field generation.
 * 
 * @param {string} userInput - User's message (used for CLOUD, not for seeding)
 * @param {object} cloudResponse - Response from CLOUD.ping()
 * @returns {object}
 */
function hazeRespond(userInput, cloudResponse) {
    fieldState.turnCount++;
    
    // 1. Get identity prefix from CLOUD's trauma analysis
    const identityPrefix = cloudResponse?.identityPrefix || 
        IDENTITY_PREFIXES[Math.floor(Math.random() * IDENTITY_PREFIXES.length)];
    
    // 2. Generate from INTERNAL FIELD (not from user input!)
    const fieldText = generateFromField(15, 0.75);
    
    // 3. Combine: identity prefix + field generation
    const rawText = identityPrefix + " " + fieldText;
    
    // 4. Cleanup
    const cleanText = cleanupText(rawText);
    
    // 5. Enrich field with user vocabulary (for future turns)
    const userWords = userInput.toLowerCase().split(/\s+/);
    for (const word of userWords) {
        if (word.length > 2) {
            fieldState.vocabulary.add(word);
            fieldState.enrichment++;
        }
    }
    
    return {
        text: cleanText,
        rawText: rawText,
        identityPrefix: identityPrefix,
        turnCount: fieldState.turnCount,
        enrichment: fieldState.enrichment,
        cloud: cloudResponse
    };
}

/**
 * Full pipeline: CLOUD → HAZE
 * @param {string} userInput - User's message
 * @returns {object}
 */
function respond(userInput) {
    // 1. CLOUD ping (pre-semantic emotion detection)
    const cloudResponse = window.CLOUD ? window.CLOUD.ping(userInput) : null;
    
    // 2. HAZE respond (NO SEED FROM PROMPT)
    const hazeResponse = hazeRespond(userInput, cloudResponse);
    
    return hazeResponse;
}

// Initialize on load
initializeField();

// Export for use in browser
window.HAZE = {
    respond,
    hazeRespond,
    generateFromField,
    cleanupText,
    initializeField,
    fieldState,
    IDENTITY_PREFIXES,
    CORPUS_FRAGMENTS
};

console.log("[HAZE] Hybrid Attention Entropy System initialized.");
console.log("[HAZE] NO SEED FROM PROMPT — speaks from internal field.");
