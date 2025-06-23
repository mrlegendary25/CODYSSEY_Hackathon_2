// Initialize chat view
document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
});

// Send message function
async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    // Disable input while processing
    messageInput.disabled = true;
    
    // Add user message to chat
    addMessage(message, true);
    messageInput.value = '';
    
    try {
        // Send message to server
        const response = await fetch(`/chat/${CHAT_SESSION_ID}/messages`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });
        
        if (!response.ok) {
            throw new Error('Failed to send message');
        }
        
        const data = await response.json();
        addMessage(data.message, false);
    } catch (error) {
        console.error('Error:', error);
        addErrorMessage();
    } finally {
        messageInput.disabled = false;
        messageInput.focus();
    }
}

// Add message to chat
function addMessage(message, isUser) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.textContent = message;
    
    // Add with fade-in animation
    messageDiv.style.opacity = '0';
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Trigger fade-in
    setTimeout(() => {
        messageDiv.style.opacity = '1';
    }, 10);
}

// Add error message
function addErrorMessage() {
    const errorMessage = 'Sorry, there was an error sending your message. Please try again.';
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message error-message';
    messageDiv.textContent = errorMessage;
    document.getElementById('chatMessages').appendChild(messageDiv);
}

// Handle enter key for sending messages
document.getElementById('messageInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Add unload warning
window.addEventListener('beforeunload', function(e) {
    if (document.getElementById('messageInput').value.trim()) {
        e.preventDefault();
        e.returnValue = '';
    }
});

async function analyzeChat() {
    try {
        let response = await fetch("/analyze_chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: CHAT_SESSION_ID })
        });
        let data = await response.json();
        if (data.success) {
            // Try to check if port 3000 is available first
            // try {
            //     const portCheck = await fetch('http://localhost:3000/health', {
            //         method: 'GET',
            //         mode: 'no-cors',
            //         timeout: 2000
            //     });
            //     window.open('http://localhost:3000', '_blank');
            // } catch (e) {
            //     console.error("Port 3000 may not be available:", e);
            //     alert("Please make sure the prediction service (port 3000) is running and try again.");
            // }
            
            // Redirect to results page regardless
            // setTimeout(() => {
            //     window.location.href = "/results";
            // }, 1000);
        } else {
            alert("Analysis failed! Try again.");
        }
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred during analysis. Please try again.");
    }
}