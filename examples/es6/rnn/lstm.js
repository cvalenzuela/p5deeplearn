/*
LSTM/RNN Generator
*/

import {Array1D, NDArrayMathGPU, Scalar} from 'deeplearn';
import { hamlet } from './hamlet';

let text = hamlet.toLowerCase();
let maxlen = 40;
let chars = new Set(Array.from(text).sort()); // \n for ↵ ?

let char_indices = {};
for (let item of chars) {
  console.log(item);
}
// let char_indices = chars.reduce(function(acc, cur, i) {
//   acc[i] = cur;
//   return acc;
// }, {});

let indices_char;

// const math = new NDArrayMathGPU();
// const a = Array1D.new([1, 2, 3]);
// const b = Scalar.new(2);
// math.scope(() => {
//   const result = math.add(a, b);
//   console.log(result.getValues());  // Float32Array([3, 4, 5])
// });


let lstm = () =>  {
  
}

export { lstm }