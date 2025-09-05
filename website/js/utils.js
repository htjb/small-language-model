export function loadHTML(elementId, filePath) {
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

export function weightedChoice(pairs) {
  let r = Math.random();
  let cumulative = 0;
  for (let [i, p] of pairs) {
    cumulative += p;
    if (r < cumulative) return i;
  }
  return pairs[pairs.length - 1][0]; // fallback
}

export async function loadYaml(url) {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
  const text = await response.text();
  return jsyaml.load(text);
}

export async function loadText(url) {
  const response = await fetch(url); // wait for the file to load
  if (!response.ok)
    throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
  const text = await response.text(); // read the text content
  return text;
}
