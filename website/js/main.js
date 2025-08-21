function loadHTML(elementId, filePath) {
    fetch(filePath)
        .then(response => response.text())
        .then(html => {
            document.getElementById(elementId).innerHTML = html;
        });
}

loadHTML('welcome-message', 'includes/welcome.html');