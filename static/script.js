function showTopicSelection() {
    document.getElementById('topicSelection').style.display = 'block';
}

function startTopicChat() {
    const topic = document.getElementById('topicSelect').value;
    startChat('topic', topic);
}

async function startChat(mode, topic = null) {
    // Hide mode selection and show chat container
    document.getElementById('modeSelection').style.display = 'none';
    document.getElementById('chatContainer').style.display = 'flex';
    
    // Initialize chat with server
    const response = await fetch('/start_chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ mode, topic })
    });
    
    const data = await response.json();
    addMessage(data.message, false);
}

function addMessage(message, isUser) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim(); //test
    
    if (!message) return;
    
    // Add user message to chat
    addMessage(message, true);
    messageInput.value = '';
    
    // Send message to server
    const response = await fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message })
    });
    
    const data = await response.json();
    addMessage(data.message, false);
}

// Allow sending message with Enter key
document.getElementById('messageInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});