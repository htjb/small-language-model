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
    }
    if (i < text.length - 1) {
      codified.push(" ");
    }
  }

  let indices = codified.map((a) => wordToIndex[a]);
  return indices;
}

function weightedChoice(pairs) {
  let r = Math.random();
  let cumulative = 0;
  for (let [i, p] of pairs) {
    cumulative += p;
    if (r < cumulative) return i;
  }
  return pairs[pairs.length - 1][0]; // fallback
}

async function runModel(initialSequence, session) {
  let input_sequence = initialSequence.map(BigInt);
  /*if (input_sequence.length < 1024) {
    input_sequence = input_sequence.concat(
      Array(1024 - input_sequence.length).fill(BigInt(0)),
    );
  }*/

  const input = new ort.Tensor("int64", new BigInt64Array(input_sequence), [
    1,
    input_sequence.length,
  ]);
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

  probs[input_sequence[initialSequence.length - 1]] = 0;

  // pair indices with probs
  let indexed_probs = probs.map((p, i) => [i, p]);

  // sort by prob
  indexed_probs.sort((a, b) => b[1] - a[1]);

  // take top-k
  let top_k = indexed_probs.slice(0, 250);

  // normalize
  let total = top_k.reduce((acc, x) => acc + x[1], 0);
  let normed = top_k.map(([i, p]) => [i, p / total]);

  let int = weightedChoice(normed);

  return { int: int, length: initialSequence.length + 1 };
}

const indexToWordContents = fs.readFileSync(
  "../model_training/classic_books_index_to_word.yaml",
  "utf8",
);
let indexToWord = yaml.load(indexToWordContents);

let text = "what is the whether";
let out = { int: "", length: 0 };

let re = /\w+|[^\w\s]/g;
let split_input = text.match(re);
split_input = split_input.map((a) => a.split(""));

let sequence = bpe(split_input);

while (out["length"] < 1025) {
  out = await runModel(sequence, session);
  sequence.push(out["int"]);
  if (indexToWord[out["int"]] === "EOS") break;
}
console.log(
  sequence
    .map((a) => indexToWord[a])
    .join("")
    .replace(/EOS/g, ""),
);
