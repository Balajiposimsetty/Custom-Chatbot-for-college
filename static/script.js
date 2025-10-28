document.addEventListener("DOMContentLoaded", function() {
    const sendButton = document.getElementById("send-button");
    sendButton.addEventListener("click", sendMessage);
    const clearButton = document.getElementById("clear-button");
    clearButton.addEventListener("click", clearMessages);
    const userInput = document.getElementById("user-input");
    userInput.addEventListener("keypress", function(event) {
        if (event.key === "Enter") sendMessage();
    });
});

function appendMessage(message, className) {
    const messageDiv = document.createElement("div");
    messageDiv.className = className;
    messageDiv.innerText = message;
    document.getElementById("messages").appendChild(messageDiv);
}

function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    if (!userInput) return;
    appendMessage(userInput, 'user-message');
    document.getElementById("user-input").value = '';
    fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ input: userInput })
    })
    .then(response => response.json())
    .then(data => {
        if (data.response) appendMessage(data.response, 'bot-message');
        else appendMessage("Sorry, I didn't understand that.", 'bot-message');
    })
    .catch(error => {
        console.error('Error:', error);
        appendMessage("There was an error communicating with the server.", 'bot-message');
    });
}

function clearMessages() {
    document.getElementById("messages").innerHTML = '';
}
