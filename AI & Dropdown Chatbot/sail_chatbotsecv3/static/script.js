// Global variables
let messages = [];
let chatMode = 'normal';
let showFeedback = false;
let showDropdown = false;
let dropdownOptions = [];
let selectedOption = '';
let isRecording = false;
let isTyping = false;
let waveformAnimationId;

// Initialize the chat interface
document.addEventListener('DOMContentLoaded', function() {
    initializeChat();
});

function initializeChat() {
    // Initial greeting with typing animation
    showTypingIndicator();

    setTimeout(() => {
        hideTypingIndicator();
        addMessage("Hello! Welcome to SAIL. How can I assist you today?", false);

        // Show initial dropdown options
        const initialOptions = [
            'About SAIL',
            'Career Opportunities',
            'FAQ',
            'PCP Knowledge Base',
            'General Inquiry',
            'Technical Support'
        ];
        showDropdownOptions(initialOptions);
    }, 1500);
}

function addMessage(text, isUser) {
    const message = {
        id: Date.now().toString(),
        text: text,
        isUser: isUser,
        timestamp: new Date()
    };

    messages.push(message);
    renderMessage(message);
    scrollToBottom();
}

function renderMessage(message) {
    const messagesArea = document.getElementById('messagesArea');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.isUser ? 'user' : 'bot'}`;
    messageDiv.style.animationDelay = `${messages.length * 0.1}s`;

    messageDiv.innerHTML = `
        <div class="avatar ${message.isUser ? 'user-avatar' : 'bot-avatar'}">
            <i class="fas fa-${message.isUser ? 'user' : 'robot'}"></i>
        </div>
        <div class="message-content">
            <div class="message-text">${message.text}</div>
            <span class="message-time">${formatTime(message.timestamp)}</span>
        </div>
    `;

    messagesArea.appendChild(messageDiv);
}

function formatTime(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function showTypingIndicator() {
    isTyping = true;
    const messagesArea = document.getElementById('messagesArea');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typingIndicator';
    typingDiv.className = 'typing-indicator';

    typingDiv.innerHTML = `
        <div class="avatar bot-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="typing-content">
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;

    messagesArea.appendChild(typingDiv);
    scrollToBottom();
}

function hideTypingIndicator() {
    isTyping = false;
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

function scrollToBottom() {
    const messagesArea = document.getElementById('messagesArea');
    messagesArea.scrollTop = messagesArea.scrollHeight;
}

function simulateTyping(responseText, callback) {
    showTypingIndicator();

    setTimeout(() => {
        hideTypingIndicator();
        addMessage(responseText, false);

        if (callback) callback();

        // Show dropdown after bot response
        const responseOptions = [
            'About SAIL',
            'Career Opportunities',
            'FAQ',
            'PCP Knowledge Base',
            'General Inquiry',
            'Technical Support'
        ];
        showDropdownOptions(responseOptions);
        showFeedbackOptions();
    }, 1500);
}

function handleSendMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();

    if (!message) return;

    addMessage(message, true);
    hideDropdownOptions();
    userInput.value = '';

    simulateTyping("Thank you for your message. How can I assist you further?");
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        handleSendMessage();
    }
}

function handleChatModeChange() {
    const select = document.getElementById('chatMode');
    chatMode = select.value;
    console.log('Chat mode changed to:', chatMode);
}

function showDropdownOptions(options) {
    dropdownOptions = options;
    const container = document.getElementById('dropdownContainer');
    const select = document.getElementById('optionsSelect');

    // Clear existing options
    select.innerHTML = '<option value="">Choose an option...</option>';

    // Add new options
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        select.appendChild(optionElement);
    });

    container.style.display = 'block';
    showDropdown = true;
}

function hideDropdownOptions() {
    const container = document.getElementById('dropdownContainer');
    container.style.display = 'none';
    showDropdown = false;
}

function handleDropdownSelect() {
    const select = document.getElementById('optionsSelect');
    selectedOption = select.value;

    if (!selectedOption) return;

    addMessage(selectedOption, true);
    hideDropdownOptions();

    simulateTyping(`You selected: ${selectedOption}. How can I help you with this?`);

    // Reset selection
    select.value = '';
    selectedOption = '';
}

function handleMicClick() {
    const micBtn = document.getElementById('micBtn');
    const recordingIndicator = document.getElementById('recordingIndicator');

    isRecording = !isRecording;

    if (isRecording) {
        micBtn.classList.add('recording');
        recordingIndicator.style.display = 'flex';
        hideDropdownOptions();
        startWaveformAnimation();

        // Simulate voice recording
        setTimeout(() => {
            addMessage("🎤 Hello, I need help with SAIL", true);
            simulateTyping("I understand you need help with SAIL. What specific information are you looking for?");

            isRecording = false;
            micBtn.classList.remove('recording');
            recordingIndicator.style.display = 'none';
            stopWaveformAnimation();
        }, 3000);
    } else {
        micBtn.classList.remove('recording');
        recordingIndicator.style.display = 'none';
        stopWaveformAnimation();
    }
}

function startWaveformAnimation() {
    const canvas = document.getElementById('waveformCanvas');
    const ctx = canvas.getContext('2d');
    let t = 0;

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Create multiple wave layers
        const waves = [
            { amplitude: 15, frequency: 0.05, color: '#3b82f6', opacity: 1 },
            { amplitude: 10, frequency: 0.08, color: '#60a5fa', opacity: 0.7 },
            { amplitude: 8, frequency: 0.12, color: '#93c5fd', opacity: 0.5 }
        ];

        waves.forEach(wave => {
            ctx.beginPath();
            ctx.strokeStyle = wave.color;
            ctx.globalAlpha = wave.opacity;
            ctx.lineWidth = 2;

            for (let x = 0; x < canvas.width; x++) {
                const y = canvas.height / 2 + wave.amplitude * Math.sin((x + t) * wave.frequency);
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }

            ctx.stroke();
        });

        ctx.globalAlpha = 1;
        t += 3;
        waveformAnimationId = requestAnimationFrame(animate);
    }

    animate();
}

function stopWaveformAnimation() {
    if (waveformAnimationId) {
        cancelAnimationFrame(waveformAnimationId);
        waveformAnimationId = null;
    }
}

function showFeedbackOptions() {
    const container = document.getElementById('feedbackContainer');
    container.style.display = 'flex';
    showFeedback = true;
}

function hideFeedbackOptions() {
    const container = document.getElementById('feedbackContainer');
    container.style.display = 'none';
    showFeedback = false;
}

function handleFeedback(positive) {
    hideFeedbackOptions();

    simulateTyping("Thank you for your feedback!", () => {
        // Additional callback if needed
    });
}

function handleLogout() {
    console.log('Logout clicked');
    alert('Logout functionality would be implemented here');
}

// Utility functions for animations and interactions
function addRippleEffect(element, event) {
    const ripple = document.createElement('span');
    const rect = element.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;

    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    ripple.classList.add('ripple');

    element.appendChild(ripple);

    setTimeout(() => {
        ripple.remove();
    }, 600);
}

// Add ripple effect to buttons
document.addEventListener('click', function(event) {
    if (event.target.matches('button, .feedback-btn, .select-btn')) {
        addRippleEffect(event.target, event);
    }
});
