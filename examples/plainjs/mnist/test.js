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

var p5_1 = new p5(function(p) {

  p.setup = function() {
    // Create DOM elements
    var canvas = p.createCanvas(280, 280);
    canvas.mousePressed(startDrawing);
    canvas.mouseReleased(stopDrawing);
    resultP = p.createP(' ');
    submit = p.createButton('classify');
    // When the button is pressed classify!
    submit.mousePressed(function(){
      classify(p)
    });
    p.background(0);
  };

  p.draw = function() {
    if (drawing) {
      p.stroke(255);
      p.strokeWeight(16);
      p.line(p.pmouseX, p.pmouseY, p.mouseX, p.mouseY);
    }
  };

  // Turn drawing on
  function startDrawing() {
    drawing = true;
    if (next) {
      // Clear the background
      p.background(0);
      next = false;
    }
  }

  // Turn drawing off when you release
  function stopDrawing() {
    drawing = false;
  }
});

var p5_2 = new p5(function(p) {

  p.setup = function() {
    var canvas = p.createCanvas(28, 28);
    p.background(0);
  };

  p.draw = function() {

  };
});


// Run the classification
function classify(p1) {
  // Get all the pixels!
  var img = p1.createImage(28, 28);
  var gray = [];
  var pixels = [];

  img.copy(p1.get(), 0, 0, p1.width, p1.height, 0, 0, 28, 28);
  img.get()
  pixels = Array.prototype.slice.call(img.pixels);

  for (var i = 0; i <= 783; i++) {
    var value = pixels.slice(0, 3).reduce(function(sum, current) {
      return sum + current
    }) / 3
    gray.push(p1.float(p1.norm(value, 0, 255).toFixed(2)))
    pixels.splice(0, 4)
  }
  console.log(gray)
  next = true
}