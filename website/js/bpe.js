import { loadYaml, loadText } from "./utils.js";

let wordToIndex = await loadYaml("./assets/classic_books_word_to_index.yaml");
let mergerRulesText = await loadText("./assets/classic_books_merger_rules.txt");

mergerRulesText = mergerRulesText.split("\n");
mergerRulesText = mergerRulesText.map((a) => a.split(" "));
mergerRulesText = mergerRulesText.map((r) => r.join(","));

export function bpe(text) {
  let vocab = Object.keys(wordToIndex);

  for (let i = 0; i < text.length; i++) {
    let j = 0;
    while (j < text[i].length) {
      let pair = [text[i][j], text[i][j + 1]];
      if (mergerRulesText.includes(pair.join(","))) {
        text[i] = [
          ...text[i].slice(0, j),
          pair.join(""),
          ...text[i].slice(j + 2, text[i].length),
        ];
        j = 0;
      } else {
        j += 1;
      }
    }
  }

  let codified = [];
  for (let i = 0; i < text.length; i++) {
    for (let subword of text[i]) {
      if ([".", "!", "?"].includes(subword)) {
        if (codified.length > 0 && codified[codified.length - 1] == " ") {
          codified.pop();
        }
        codified.push(subword);
        codified.push("EOS");
      } else if (vocab.includes(subword)) {
        codified.push(subword);
      } else {
        codified.push("UNK");
      }
    }
    if (i < text.length - 1) {
      codified.push(" ");
    }
  }

  console.log(codified);

  let indices = codified.map((a) => wordToIndex[a]);
  return indices;
}
