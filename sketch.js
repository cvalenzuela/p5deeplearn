var reader = new deeplearn.CheckpointLoader('.');
var data, math, _a,sess, input, probs, canvas;
var drawing = false;

var smallCanvas;
var largeImg;
var resultText = document.getElementById('result');

function setup(){
  canvas = createCanvas(280, 280);
  canvas.parent('canvas');
  pixelDensity(1);
  noStroke();
  fill(255);
  background(0);

  smallCanvas = createGraphics(28,28);
  smallCanvas.background(0);
  largeImg = createImage(280,280, RGB);
}

function draw(){
  
  if (mouseIsPressed && mouseX < 280 && mouseY < 280 && mouseX > 0 && mouseY > 0){
    drawing = true;
    ellipse(mouseX, mouseY, 25, 25);
  } 
  if(!drawing){
    image(largeImg, 0, 0, 280, 280);
  }
  
}

function clearCanvas(){
  drawing = false;
  smallCanvas.loadPixels();
  for(var y = 0; y < smallCanvas.height; y++){
    for(var x = 0; x < smallCanvas.width; x++){
      smallCanvas.set(x,y, 0)
    }
  }
  smallCanvas.updatePixels();
  copyCanvas();
}

function showRandomNumber(){
  drawing = false;
  smallCanvas.background(255);
  smallCanvas.loadPixels();
  var numberToGuess = Math.floor(Math.random() * 50);
  var index = 0;
  for(var y = 0; y < smallCanvas.height; y++){
    for(var x = 0; x < smallCanvas.width; x++){
      smallCanvas.set(x,y, data.images[numberToGuess][index]*255)
      index ++;
    }
  }
  smallCanvas.updatePixels();
  copyCanvas();
}

function copyCanvas(){
  var img = smallCanvas.get();
  largeImg.copy(img, 0, 0, 28, 28, 0,0, 280,280);
}

function getNumber(){
  if(drawing){
    var smallImg = smallCanvas.createImage(28, 28, RGB);
    smallImg.copy(get(), 0, 0, 280, 280, 0,0, 28,28);
    smallCanvas.image(smallImg, 0, 0, 28, 28);
  }
  var imageArray = [];
  var normalizedArray = [];
  var index = 0;

  for(var y = 0; y < smallCanvas.height; y++){
    for(var x = 0; x < smallCanvas.width; x++){
      imageArray.push(smallCanvas.get(x,y))
      index ++;
    }
  }
  
  for (var i = 0; i < imageArray.length; i++){
    var value = norm(imageArray[i][0], 0, 255);
    (value > 0) && (value = float(value.toFixed(2)));
      normalizedArray.push(value);
  }

  recognizeData(normalizedArray);
}

function recognizeData (inputToClassify) {
  math.scope(function () {
    var inputData = deeplearn.Array1D.new(inputToClassify);
    var probsVal = sess.eval(probs, [{ tensor: input, data: inputData }]);
    resultText.innerHTML = 'Prediction: ' + probsVal.get();
    console.log('Prediction: ' + probsVal.get());
  });
};

reader.getAllVariables().then(function (vars) {
  var xhr = new XMLHttpRequest();
  xhr.open('GET', 'sample_data.json');
  xhr.onload = function () {
    data = JSON.parse(xhr.responseText);
    math = new deeplearn.NDArrayMathGPU();
    _a = buildModelGraphAPI(data, vars);
    input = _a[0];
    probs = _a[1];
    sess = new deeplearn.Session(input.node.graph, math);
              
    // math.scope(function () {
    //   var numCorrect = 0;
    //   for (var i = 0; i < data.images.length; i++) {
    //     var inputData = deeplearn.Array1D.new(data.images[i]);
    //     var probsVal = sess.eval(probs, [{ tensor: input, data: inputData }]);
    //     if (data.labels[i] === probsVal.get()) {
    //       numCorrect++;
    //     }
    //   }
    // });
  };

  xhr.onerror = function (err) { return console.error(err); };
  xhr.send();
});

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

