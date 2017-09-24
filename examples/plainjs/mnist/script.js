/*
===
Mnist Demo
p5 and deeplearn.js

This is a port of Daniel Shiffman Nature of Code: Intelligence and Learning
Original Repo: https://github.com/shiffman/NOC-S17-2-Intelligence-Learning

Cristóbal Valenzuela
https://github.com/cvalenzuela/p5deeplearn
===
*/

var reader = new deeplearn.CheckpointLoader('.');
var data, math, _a, sess, input, probs;

var submit; // Submit button
var resultP; // Show results

var next = false;
var drawing = false;

function setup() {
  // Create DOM elements
  var canvas = createCanvas(280, 280);
  canvas.mousePressed(startDrawing);
  canvas.mouseReleased(stopDrawing);
  resultP = createP(' ');
  submit = createButton('classify');
  // When the button is pressed classify!
  submit.mousePressed(classify);
  background(0);  
}

// Turn drawing on
function startDrawing() {
  drawing = true;
  if (next) {
    // Clear the background
    background(0);
    next = false;
  }
}

// Turn drawing off when you release
function stopDrawing() {
  drawing = false;
}

function draw() {
  // If you are drawing
  if (drawing) {
    stroke(255);
    strokeWeight(16);
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
}

// Run the classification
function classify() {
  // Get all the pixels!
  var img = createImage(28, 28);
  var gray = [];
  var pixels = [];

  img.copy(get(), 0, 0, width, height, 0, 0, 28, 28);
  img.get()
  pixels = Array.prototype.slice.call(img.pixels);

  for (var i = 0; i <= 783; i++) {
    var value = pixels.slice(0, 3).reduce(function(sum, current) {
      return sum + current
    }) / 3
    gray.push(float(norm(value, 0, 255).toFixed(2)))
    pixels.splice(0, 4)
  }
  console.log(gray)
  next = true

  //predict(gray)
}

function predict(data) {
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
      resultP.html(probsVal.get());
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