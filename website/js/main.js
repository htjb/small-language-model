import { loadHTML } from "./utils.js";
import { callModel } from "./load.js";

function getInputText() {
  if (welcomeMessage.innerHTML) {
    welcomeMessage.innerHTML = "";
  }
  history += " ";
  history += input.value;

  let inputDiv = document.createElement("div");
  inputDiv.setAttribute("class", "input-element");
  inputDiv.textContent = input.value;
  document.getElementById("chat-history").appendChild(inputDiv);

  callModel(history).then((result) => {
    let outputDiv = document.createElement("div");
    outputDiv.setAttribute("class", "output-element");
    outputDiv.textContent = result;
    history += result;
    document.getElementById("chat-history").appendChild(outputDiv);
  });

  input.value = "";
}

loadHTML("welcome-message", "includes/welcome.html");

let history = "";
let input = document.getElementById("user-input");
let submit = document.getElementById("submit-button");
let welcomeMessage = document.getElementById("welcome-message");

submit.addEventListener("click", getInputText);
input.addEventListener("keydown", function (e) {
  if (e.keyCode == 13) {
    getInputText();
  }
});
