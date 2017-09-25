/*
Experimental Wrapper for deeplearn.js
*/

var data, math, _a, sess, input, probs;

function predict(data) {
  var reader = new deeplearn.CheckpointLoader('.');

  reader.getAllVariables().then(function(vars) {
    math = new deeplearn.NDArrayMathGPU();
    _a = buildModelGraphAPI(data, vars);
    
    input = _a[0];
    probs = _a[1];
    sess = new deeplearn.Session(input.node.graph, math);
    
    math.scope(function() {
      var inputData = deeplearn.Array1D.new(data);
      var probsVal = sess.eval(probs, [{ tensor: input, data: inputData }]);
      console.log('Prediction: ' + probsVal.get());
      resultP.html('Prediction: ' + probsVal.get());
      sess.dispose();
    });
  });
};

function buildModelGraphAPI(data, vars) {
  var g = new deeplearn.Graph();
  var input = g.placeholder('input', [784]);
  var hidden1W = g.constant(vars['hidden1/weights']);
  var hidden1B = g.constant(vars['hidden1/biases']);
  var hidden1 = g.relu(g.add(g.matmul(input, hidden1W), hidden1B));
  var hidden2W = g.constant(vars['hidden2/weights']);
  var hidden2B = g.constant(vars['hidden2/biases']);
  var hidden2 = g.relu(g.add(g.matmul(hidden1, hidden2W), hidden2B));
  var softmaxW = g.constant(vars['softmax_linear/weights']);
  var softmaxB = g.constant(vars['softmax_linear/biases']);
  var logits = g.add(g.matmul(hidden2, softmaxW), softmaxB);
  return [input, g.argmax(logits)];
}