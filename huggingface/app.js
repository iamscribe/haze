/*
 * app.js — HAZE Chat Application
 * 
 * Connects the UI with CLOUD + HAZE pipeline.
 * 
 * KEY: NO SEED FROM PROMPT
 * User input goes to CLOUD (emotion detection),
 * but HAZE generates from its internal field.
 */

document.addEventListener('DOMContentLoaded', () => {
    const messagesContainer = document.getElementById('messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const cloudIndicator = document.getElementById('cloud-indicator');
    
    // Stats panel
    const cloudStatus = document.getElementById('cloud-status');
    const primaryEmotion = document.getElementById('primary-emotion');
    const secondaryEmotion = document.getElementById('secondary-emotion');
    const iterationsDisplay = document.getElementById('iterations');
    
    /**
     * Add message to chat
     */
    function addMessage(type, content, meta = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}`;
        
        const prefix = document.createElement('span');
        prefix.className = 'prefix';
        prefix.textContent = type === 'user' ? '[you]' : type === 'haze' ? '[haze]' : '[system]';
        
        const contentSpan = document.createElement('span');
        contentSpan.className = 'content';
        contentSpan.textContent = content;
        
        msgDiv.appendChild(prefix);
        msgDiv.appendChild(contentSpan);
        
        // Add cloud info for haze messages
        if (type === 'haze' && meta?.cloud) {
            const cloudInfo = document.createElement('div');
            cloudInfo.className = 'cloud-info';
            cloudInfo.innerHTML = `
                <span class="emotion-tag">${meta.cloud.primary}</span>
                <span class="secondary-tag">+ ${meta.cloud.secondary}</span>
                <span class="iterations-tag">${meta.cloud.iterations} iter</span>
            `;
            msgDiv.appendChild(cloudInfo);
        }
        
        messagesContainer.appendChild(msgDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    /**
     * Update stats panel
     */
    function updateStats(response) {
        if (response.cloud) {
            cloudStatus.textContent = 'active';
            cloudStatus.className = 'value active';
            primaryEmotion.textContent = response.cloud.primary;
            secondaryEmotion.textContent = response.cloud.secondary;
            iterationsDisplay.textContent = response.cloud.iterations;
            
            // Update chamber bars if they exist
            if (response.cloud.chambers) {
                updateChamberBars(response.cloud.chambers);
            }
        }
    }
    
    /**
     * Update chamber visualization
     */
    function updateChamberBars(chambers) {
        const existingBars = document.getElementById('chamber-bars');
        if (existingBars) {
            existingBars.remove();
        }
        
        const barsDiv = document.createElement('div');
        barsDiv.id = 'chamber-bars';
        barsDiv.className = 'chamber-bars';
        
        for (const [chamber, value] of Object.entries(chambers)) {
            const barContainer = document.createElement('div');
            barContainer.className = 'chamber-bar-container';
            
            const label = document.createElement('span');
            label.className = 'chamber-label';
            label.textContent = chamber.substring(0, 4);
            
            const bar = document.createElement('div');
            bar.className = `chamber-bar ${chamber.toLowerCase()}`;
            bar.style.width = `${value * 100}%`;
            
            barContainer.appendChild(label);
            barContainer.appendChild(bar);
            barsDiv.appendChild(barContainer);
        }
        
        document.getElementById('stats-panel').appendChild(barsDiv);
    }
    
    /**
     * Handle user message
     */
    function handleSend() {
        const text = userInput.value.trim();
        if (!text) return;
        
        // Add user message
        addMessage('user', text);
        userInput.value = '';
        
        // Get HAZE response (with CLOUD)
        setTimeout(() => {
            try {
                const response = window.HAZE.respond(text);
                addMessage('haze', response.text, response);
                updateStats(response);
            } catch (e) {
                addMessage('system', `Error: ${e.message}`);
            }
        }, 100); // Small delay for visual feedback
    }
    
    // Event listeners
    sendBtn.addEventListener('click', handleSend);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSend();
        }
    });
    
    // Focus input
    userInput.focus();
    
    console.log("[APP] HAZE chat interface ready.");
    console.log("[APP] NO SEED FROM PROMPT — HAZE speaks from internal field.");
});
