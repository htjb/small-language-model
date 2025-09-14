import { weightedChoice, loadYaml } from "./utils.js";
import { bpe } from "./bpe.js";

let indexToWord = await loadYaml("./assets/classic_books_index_to_word.yaml");

async function runModel(initialSequence, session) {
  let inputSequence = initialSequence.map(BigInt);

  const input = new ort.Tensor("int64", new BigInt64Array(inputSequence), [
    1,
    inputSequence.length,
  ]);
  const feeds = { x: input }; // change input_name accordingly

  const results = await session.run(feeds);
  let res = results["output"]["cpuData"];
  res[0] = -Infinity;

  let sumExpResult = 0;
  let expResult = [];
  for (let i = 0; i < res.length; i++) {
    expResult.push(Math.exp(res[i]));
    sumExpResult += Math.exp(res[i]);
  }

  let probs = [];
  for (let er of expResult) {
    probs.push(er / sumExpResult);
  }

  probs[inputSequence[initialSequence.length - 1]] = 0;

  // pair indices with probs
  let indexProbs = probs.map((p, i) => [i, p]);

  // sort by prob
  indexProbs.sort((a, b) => b[1] - a[1]);

  // take top-k
  let topK = indexProbs.slice(0, 250);

  // normalize
  let total = topK.reduce((acc, x) => acc + x[1], 0);
  let normed = topK.map(([i, p]) => [i, p / total]);

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
  let splitInput = text.match(re);
  splitInput = splitInput.map((a) => a.split(""));

  let sequence = bpe(splitInput);
  let originalSequenceLength = sequence.length;
  while (out["length"] < 1025) {
    out = await runModel(sequence, session);
    sequence.push(out["int"]);
    if (indexToWord[out["int"]] === "EOS") break;
  }

  console.log(originalSequenceLength, sequence.length);
  let output = sequence
    .slice(originalSequenceLength, sequence.length)
    .map((a) => indexToWord[a])
    .join("")
    .replace(/EOS/g, "");

  return output;
}
