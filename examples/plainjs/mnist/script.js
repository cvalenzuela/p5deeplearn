/*
===
Mnist Demo
deeplearn.js meets p5

This is a port of Daniel Shiffman Nature of Code: Intelligence and Learning
Original Repo: https://github.com/shiffman/NOC-S17-2-Intelligence-Learning

Cristóbal Valenzuela
https://github.com/cvalenzuela/p5deeplearn
===
*/

var submit; // Submit button
var resultP; // Show results

var next = false;
var drawing = false;

function setup() {
  // Create DOM elements
  var canvas = createCanvas(280, 280);
  pixelDensity(1);
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
    stroke(235);
    strokeWeight(30);
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
}

// Run the classification
function classify() {
  var img = createImage(28, 28);
  var gray = [];

  img.copy(get(), 0, 0, width, height, 0, 0, 28, 28);
  img.loadPixels();
  var imgPixels = Array.prototype.slice.call(img.pixels);

  for (var i = 0; i < 784; i++) {
    var value = imgPixels.slice(0, 3).reduce(function(sum, current) {
      return sum + current
    }) / 3
    gray.push(float(norm(value, 0, 255).toFixed(3)))
    imgPixels.splice(0, 4)
  }
  next = true;  
  
  predict(gray);  

  // Debug: Draw the greyscale 28x28 image in the corner
  //copy(img.get(),0,0,28,28,0,0,28,28)  

}