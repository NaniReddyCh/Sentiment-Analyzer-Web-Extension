document.getElementById('submitFeedback').addEventListener('click', function() {
    const feedback = document.getElementById('feedback').value;
    const url = window.location.href;

    if (feedback.length === 0) {
        document.getElementById('message').innerText = 'Please enter your feedback.';
        return;
    }

    fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ feedback: feedback, url: url })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('message').innerText = data.message;
    });
});
