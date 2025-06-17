// Main application script
document.addEventListener('DOMContentLoaded', function() {
    // ===== Page Detection =====
    const isResultsPage = document.getElementById('prediction-data') !== null;
    const isHomePage = document.querySelector('.upload-area') !== null;

    // ===== Results Page Functionality =====
    if (isResultsPage) {
        const chatContainer = document.getElementById('chat-container');
        const questionInput = document.getElementById('ai-question');
        const askBtn = document.getElementById('ask-ai-btn');
        const predictionData = {
            class: document.getElementById('prediction-data').dataset.class,
            class_id: parseInt(document.getElementById('prediction-data').dataset.classId),
            confidence: parseFloat(document.getElementById('prediction-data').dataset.confidence),
            probabilities: JSON.parse(document.getElementById('prediction-data').dataset.probabilities)
        };

        function addChatMessage(sender, message, type = 'ai') {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function sendAiQuestion() {
            const question = questionInput.value.trim();
            if (!question) return;
            
            addChatMessage('You', question, 'user');
            questionInput.value = '';
            questionInput.disabled = true;
            askBtn.disabled = true;
            
            fetch('/ask_ai', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    prediction: predictionData
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    addChatMessage('AI', `Error: ${data.error}`, 'error');
                } else {
                    addChatMessage('AI', data.response, 'ai');
                }
            })
            .catch(error => {
                addChatMessage('AI', 'Error connecting to the AI service', 'error');
                console.error('Error:', error);
            })
            .finally(() => {
                questionInput.disabled = false;
                askBtn.disabled = false;
                questionInput.focus();
            });
        }

        askBtn.addEventListener('click', sendAiQuestion);
        questionInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendAiQuestion();
        });
    }

    // ===== Home Page Functionality =====
    if (isHomePage) {
        const dropArea = document.getElementById('drop-area');
        const fileInput = document.getElementById('file-input');
        const preview = document.getElementById('preview');
        const submitBtn = document.getElementById('submit-btn');

        // Existing home page functionality...
        dropArea.addEventListener('click', () => fileInput.click());
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });

        function highlight() {
            dropArea.style.borderColor = '#764ba2';
            dropArea.style.backgroundColor = '#f0f0f0';
        }

        function unhighlight() {
            dropArea.style.borderColor = '#ccc';
            dropArea.style.backgroundColor = '';
        }

        dropArea.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            fileInput.files = files;
            handleFiles(files);
        }

        fileInput.addEventListener('change', () => {
            handleFiles(fileInput.files);
        });

        function handleFiles(files) {
            if (files.length > 0) {
                const file = files[0];
                if (file.type.match('image.*')) {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                        submitBtn.disabled = false;
                    };
                    reader.readAsDataURL(file);
                }
            }
        }
    }

    // ===== Global Functionality =====
    // Navigation items
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            navItems.forEach(i => i.classList.remove('active'));
            this.classList.add('active');
        });
    });
});