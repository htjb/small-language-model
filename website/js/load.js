import { weightedChoice, loadYaml } from "./utils.js";
import { bpe } from "./bpe.js";

let indexToWord = await loadYaml("./assets/classic_books_index_to_word.yaml");

async function runModel(initialSequence, session) {
  let input_sequence = initialSequence.map(BigInt);

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

export async function callModel(text) {
  const session = await ort.InferenceSession.create(
    "./assets/classic_books_model.onnx",
    {
      executionProviders: ["wasm"],
    },
  );

  let out = { int: "", length: 0 };

  let re = /\w+|[^\w\s]/g;
  let split_input = text.match(re);
  split_input = split_input.map((a) => a.split(""));

  let sequence = bpe(split_input);
  let original_sequence_length = sequence.length;
  while (out["length"] < 1025) {
    out = await runModel(sequence, session);
    sequence.push(out["int"]);
    if (indexToWord[out["int"]] === "EOS") break;
  }

  console.log(original_sequence_length, sequence.length);
  let output = sequence
    .slice(original_sequence_length, sequence.length)
    .map((a) => indexToWord[a])
    .join("")
    .replace(/EOS/g, "");

  return output;
}
