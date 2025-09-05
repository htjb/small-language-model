import { loadHTML } from "./utils.js";

function getInputText() {
  if (welcomeMessage.innerHTML) {
    welcomeMessage.innerHTML = "";
  }
  history += input.value;
  console.log(history);
  let inputDiv = document.createElement("div");
  inputDiv.setAttribute("class", "input-element");
  inputDiv.textContent = input.value;
  document.getElementById("chat-history").appendChild(inputDiv);
}

loadHTML("welcome-message", "includes/welcome.html");

let history = "";
let input = document.getElementById("user-input");
let submit = document.getElementById("submit-button");
let welcomeMessage = document.getElementById("welcome-message");

submit.addEventListener("click", getInputText);
input.addEventListener("keydown", function (e) {
  if (e.keyCode == 13) {
    console.log("enter pressed");
    getInputText();
  }
});
