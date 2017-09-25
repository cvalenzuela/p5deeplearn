/*
LSTM/RNN Generator
*/

import { Array3D, NDArrayMathGPU, CheckpointLoader, Session } from 'deeplearn';
import { hamlet } from './hamlet';

let math, _a;

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

let reader = new CheckpointLoader('.');

reader.getAllVariables().then((vars) => {
  math = new NDArrayMathGPU();
  _a = buildModelGraphAPI(data, vars);

  input = _a[0];
  probs = _a[1];
  sess = new Session(input.node.graph, math);

  math.scope(() => {
    let inputData = track(Array1D.new(data));
    let probsVal = sess.eval(probs, [{
      tensor: input,
      data: inputData
    }]);
    console.log('Prediction: ' + probsVal.get());
    resultTag.html('Prediction: ' + probsVal.get());
    sess.dispose();
  });
});

let lstm = () =>  {
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

lstm();

export { lstm }