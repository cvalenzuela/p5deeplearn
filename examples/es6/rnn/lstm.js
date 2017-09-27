/*
LSTM/RNN Generator
*/

import { Array3D, NDArrayMathGPU, CheckpointLoader, Session, Graph } from 'deeplearn';
import { hamlet } from './hamlet';

let input, probs, session;

const math = new NDArrayMathGPU();

let reader = new CheckpointLoader('model/');
reader.getAllVariables().then((checkpoints) => {
  let graphModel = buildModelGraph(checkpoints);
  // input = graphModel[0];
  // probs = graphModel[1];
  // console.log(input)
  // session = new Session(input.node.graph, math);
});

let getRandomInt = (min, max) => {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

let text = hamlet.toLowerCase();
let maxlen = 40;
let chars = Array.from(new Set(Array.from(text))).sort(); // \n for ↵ ?
let char_indices = chars.reduce((acc, cur, i) => {
  acc[cur] = i;
  return acc;
}, {});
let indices_char = chars.reduce((acc, cur, i) => {
  acc[i] = cur;
  return acc;
}, {});

let start_index = getRandomInt(0, text.length - maxlen - 1);
let diversity = 0.5;
let generated = '';
let sentence = text.substring(start_index, start_index + maxlen);
generated += sentence;

let buildModelGraph = (checkpoints) => {
  console.log('here')
  // let g = new Graph();
  // let input = g.placeholder('input', [784]);
  // let hidden1W = g.constant(checkpoints['hidden1/weights']);
  // let hidden1B = g.constant(checkpoints['hidden1/biases']);
  // let hidden1 = g.relu(g.add(g.matmul(input, hidden1W), hidden1B));
  // let hidden2W = g.constant(checkpoints['hidden2/weights']);
  // let hidden2B = g.constant(checkpoints['hidden2/biases']);
  // let hidden2 = g.relu(g.add(g.matmul(hidden1, hidden2W), hidden2B));
  // let softmaxW = g.constant(checkpoints['softmax_linear/weights']);
  // let softmaxB = g.constant(checkpoints['softmax_linear/biases']);
  // let logits = g.add(g.matmul(hidden2, softmaxW), softmaxB);
  // return [input, g.argmax(logits)];
};

let generateText = () =>  {
  for (let i = 0; i < 50; i++) {
    let x = Array3D.zeros([1, maxlen, chars.length]);
    Array.from(sentence).forEach((char, i) => {
      x[0, i, char_indices[char]] = 1.
    })
    let preds = sess.eval(probs, [{
      tensor: input,
      data: inputData
    }]);
  }
}

export { generateText }