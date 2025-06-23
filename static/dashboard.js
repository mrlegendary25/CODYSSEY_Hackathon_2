function showTopicSelection() {
    document.getElementById('topicSelection').style.display = 'block';
}

async function startNewChat(mode) {
    let topic = null;
    if (mode === 'topic') {
        topic = document.getElementById('topicSelect').value;
    }

    const response = await fetch('/start_chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ mode, topic })
    });

    const data = await response.json();
    if (data.session_id) {
        window.location.href = `/chat/${data.session_id}`;
    }
}

function openChat(chatId) {
    window.location.href = `/chat/${chatId}`;
}

async function deleteChat(chatId, event) {
    event.stopPropagation();
    
    if (!confirm('Are you sure you want to delete this chat?')) {
        return;
    }

    const response = await fetch(`/delete_chat/${chatId}`, {
        method: 'DELETE'
    });

    if (response.ok) {
        // Remove the chat item from the DOM
        const chatItem = event.target.closest('.chat-item');
        chatItem.remove();
    }
}

// Add this to dashboard.js
// async function showGraphs(chatId, event) {
//     event.stopPropagation();
//     const button = event.currentTarget;
//     const buttonText = button.querySelector('.btn-text');
//     const spinner = button.querySelector('.loading-spinner');
    
//     // Show loading state
//     buttonText.style.display = 'none';
//     spinner.style.display = 'block';
    
//     try {
//         const response = await fetch(`/generate_graphs/${chatId}`, {
//             method: 'GET',
//             headers: {
//                 'Accept': 'application/json',
//                 'X-Requested-With': 'XMLHttpRequest'
//             }
//         });
        
//         if (!response.ok) {
//             throw new Error(`HTTP error! status: ${response.status}`);
//         }
        
//         const data = await response.json();
async function showGraphs(chatId, event) {
    event.stopPropagation();
    const button = event.currentTarget;
    const buttonText = button.querySelector('.btn-text');
    const spinner = button.querySelector('.loading-spinner');
    
    // Show loading state
    buttonText.style.display = 'none';
    spinner.style.display = 'block';
    
    try {
        console.log(`Making request to /generate_graphs/${chatId}`);
        const response = await fetch(`/generate_graphs/${chatId}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            }
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Error response:', errorText);
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }
        
        const data = await response.json();
        console.log('Received data:', data);
        
        // Rest of your code...
        
        // Get the modal and graph containers
        const modal = document.getElementById('graphModal');
        const goEmotionsContainer = document.getElementById('goEmotionsGraph');
        const ekmanContainer = document.getElementById('ekmanGraph');
        
        // Clear previous charts if they exist
        if (goEmotionsContainer.chart) {
            goEmotionsContainer.chart.destroy();
        }
        if (ekmanContainer.chart) {
            ekmanContainer.chart.destroy();
        }
        
        // Create GoEmotions graph
        goEmotionsContainer.chart = new Chart(goEmotionsContainer, {
            type: 'bar',
            data: {
                labels: Object.keys(data.goemotions),
                datasets: [{
                    label: 'Probability',
                    data: Object.values(data.goemotions),
                    backgroundColor: 'rgba(54, 162, 235, 0.8)'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
        
        // Create Ekman graph
        ekmanContainer.chart = new Chart(ekmanContainer, {
            type: 'bar',
            data: {
                labels: Object.keys(data.ekman),
                datasets: [{
                    label: 'Probability',
                    data: Object.values(data.ekman),
                    backgroundColor: 'rgba(75, 192, 192, 0.8)'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
        
        modal.style.display = 'block';
    } catch (error) {
        console.error('Error generating graphs:', error);
        alert('Error generating graphs. Please try again later.');
    } finally {
        // Restore button state
        buttonText.style.display = 'block';
        spinner.style.display = 'none';
    }
}

// Add modal close functionality
// document.querySelector('.close-modal').addEventListener('click', () => {
//     document.getElementById('graphModal').style.display = 'none';
// });

// // Close modal when clicking outside
// window.addEventListener('click', (event) => {
//     const modal = document.getElementById('graphModal');
//     if (event.target === modal) {
//         modal.style.display = 'none';
//     }
// });

document.addEventListener('DOMContentLoaded', function() {
    // Initialize modal elements
    const modal = document.getElementById('graphModal');
    const closeButton = document.querySelector('.close-modal');
    
    if (closeButton) {
        closeButton.addEventListener('click', () => {
            modal.style.display = 'none';
        });
    }

    // Close modal when clicking outside
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
});