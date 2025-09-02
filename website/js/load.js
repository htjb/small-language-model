import * as ort from "onnxruntime-node";
import * as fs from "fs";
import * as yaml from "js-yaml";

const session = await ort.InferenceSession.create(
  "../model_training/classic_books_model.onnx",
);

function bpe(text) {
  let merger_rules = fs.readFileSync(
    "../model_training/classic_books_merger_rules.txt",
    "utf8",
  );

  const wordToIndexContents = fs.readFileSync(
    "../model_training/classic_books_word_to_index.yaml",
    "utf8",
  );
  let wordToIndex = yaml.load(wordToIndexContents);
  let vocab = Object.keys(wordToIndex);

  merger_rules = merger_rules.split("\n");
  merger_rules = merger_rules.map((a) => a.split(" "));
  merger_rules = merger_rules.map((r) => r.join(","));

  for (let i = 0; i < text.length; i++) {
    let j = 0;
    while (j < text[i].length) {
      let pair = [text[i][j], text[i][j + 1]];
      if (merger_rules.includes(pair.join(","))) {
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
      if (i < text.length - 1) {
        codified.push(" ");
      }
    }
  }

  let indices = codified.map((a) => wordToIndex[a]);
  return indices;
}

function weightedChoice(items, weights) {
  const r = Math.random();

  let cumulative = 0;
  for (let i = 0; i < items.length; i++) {
    cumulative += weights[i];
    if (r < cumulative) {
      return items[i];
    }
  }
}

async function runModel(text, session) {
  let re = /\w+|[^\w\s]/g;
  let split_input = text.match(re);
  split_input = split_input.map((a) => a.split(""));

  let prepadded_input_sequence = bpe(split_input);
  let input_sequence = prepadded_input_sequence.map(BigInt);
  if (input_sequence.length < 1024) {
    input_sequence = input_sequence.concat(
      Array(1024 - input_sequence.length).fill(BigInt(0)),
    );
  }

  const input = new ort.Tensor(
    "int64",
    new BigInt64Array(input_sequence),
    [1, 1024],
  );
  const feeds = { x: input }; // change input_name accordingly

  const results = await session.run(feeds);
  let res = results["output"]["cpuData"];
  res[0] = -Infinity;

  let sum_exp_result = 0;
  let exp_result = [];
  for (let i = 0; i < res.length; i++) {
    exp_result.push(Math.exp(res[i]));
    sum_exp_result += Math.exp(res[i]);
  }

  let probs = [];
  for (let er of exp_result) {
    probs.push(er / sum_exp_result);
  }

  probs[prepadded_input_sequence[prepadded_input_sequence.length - 1]] = 0;

  let indices = probs
    .map((value, index) => [index, value])
    .sort((a, b) => b[1] - a[1]) // sort by value descending
    .slice(0, 250) // take top k
    .map((pair) => pair[0]); // extract indices

  // now do top k
  probs.sort((a, b) => b - a);
  let top_k = probs.slice(0, 250);
  let top_k_sum = top_k.reduce((a, b) => a + b);
  top_k = top_k.map((a) => a / top_k_sum);

  let int = weightedChoice(indices, top_k);

  return { int: int, length: prepadded_input_sequence.length + 1 };
}

const indexToWordContents = fs.readFileSync(
  "../model_training/classic_books_index_to_word.yaml",
  "utf8",
);
let indexToWord = yaml.load(indexToWordContents);

let text = "alice was beginning to what ";
let out = { int: "", length: 0 };

while (out["length"] < 1025) {
  out = await runModel(text, session);
  text = text + indexToWord[out["int"]];
  if (indexToWord[out["int"]] === "EOS") break;
}
console.log(text);
