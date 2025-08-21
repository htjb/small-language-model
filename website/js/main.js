function loadHTML(elementId, filePath) {
  /* 
    Look for the file.
    If file exists extract contents as text.
    Fill HTML element with text.
    */
  fetch(filePath)
    .then((response) => response.text())
    .then((html) => {
      document.getElementById(elementId).innerHTML = html;
    });
}

loadHTML("welcome-message", "includes/welcome.html");
