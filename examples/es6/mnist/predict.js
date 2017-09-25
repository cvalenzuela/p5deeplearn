/*
Predict a drawn number.
*/

import { CheckpointLoader, NDArrayMathGPU, Session, Graph, Array1D } from 'deeplearn';

let _a, sess, input, probs, math;

let predict = (data, resultTag ) => {
  let reader = new CheckpointLoader('.');

  reader.getAllVariables().then((vars) => {
    math = new NDArrayMathGPU();
    _a = buildModelGraphAPI(data, vars);

    input = _a[0];
    probs = _a[1];
    sess = new Session(input.node.graph, math);

    math.scope(() => {
      let inputData = Array1D.new(data);
      let probsVal = sess.eval(probs, [{
        tensor: input,
        data: inputData
      }]);
      console.log('Prediction: ' + probsVal.get());
      resultTag.html('Prediction: ' + probsVal.get());
      sess.dispose();
    });
  });
};

let buildModelGraphAPI = (data, vars) => {
  let g = new Graph();
  let input = g.placeholder('input', [784]);
  let hidden1W = g.constant(vars['hidden1/weights']);
  let hidden1B = g.constant(vars['hidden1/biases']);
  let hidden1 = g.relu(g.add(g.matmul(input, hidden1W), hidden1B));
  let hidden2W = g.constant(vars['hidden2/weights']);
  let hidden2B = g.constant(vars['hidden2/biases']);
  let hidden2 = g.relu(g.add(g.matmul(hidden1, hidden2W), hidden2B));
  let softmaxW = g.constant(vars['softmax_linear/weights']);
  let softmaxB = g.constant(vars['softmax_linear/biases']);
  let logits = g.add(g.matmul(hidden2, softmaxW), softmaxB);
  return [input, g.argmax(logits)];
};

export { predict };