import imagenetClasses from './imagenetClasses.js';

const onnx = window.onnx;

const video = document.querySelector('video')
const canvas = document.querySelector('canvas');
const button = document.querySelector('button');
canvas.width = 224;
canvas.height = 224;

const modelURL = "https://raw.githubusercontent.com/Microsoft/onnxjs-demo/data/data/examples/models/squeezenetV1_8.onnx";
const session = new onnx.InferenceSession();
session.loadModel(modelURL).then(() => {console.log("model loaded");});

const classProbsList = document.getElementById("classProbsList");

function setUpClassProbsList() {
  Object.keys(imagenetClasses).map(function(key, index) {
    // console.log(index);
    // console.log(key);
    // console.log(imagenetClasses[key][0]);
    // console.log(imagenetClasses[key][1]);
    var node = document.createElement("li");
    node.setAttribute("id", imagenetClasses[key][0]);
    var textnode = document.createTextNode("");
    node.appendChild(textnode);
  classProbsList.appendChild(node);
  });
}

function drawVideoOnCanvas(video, canvas) {
  console.log("in drawVideoOnCanvas");
  // console.log(video.videoWidth);
  // canvas.width = video.videoWidth;
  // canvas.height = video.videoHeight;
  console.log(["canvas", canvas.width, canvas.height]);
  console.log(["video", video.width, video.height]);
  const context = canvas.getContext('2d')
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
}

function getArrayFromCanvasOld(canvas) {
  console.log("in getArrayFromCanvas");
  const context = canvas.getContext('2d');
  console.log(context);
  const dataUint8 = context.getImageData(0, 0, canvas.width, canvas.height).data;
  return dataUint8;
}

function getTensorFromArray(data) {
  var pixels = [];
  for (let d = 0; d < 3; d++) {
    for (let i = 0; i < data.length; i+=4) {
      pixels.push((data[i+d] - 127.5)/127.5);
      // pixels.push(data[i+d] * 1.0);
    }
  }
  const float32Array = new Float32Array(pixels);
  const tensor = new onnx.Tensor(float32Array, 'float32', [1, 3, 224, 224]);
  // is this tensor shaped correctly?
  var [c,y,x, message] = [1,0,0,"top left"];
  console.log([`${message}: c=${c}, y=${y}, x=${x}`, data[c + y*224*4 + x*4], tensor.get([0, c, y, x])]);
  var [c,y,x, message] = [1,223,0,"bottom left"];
  console.log([`${message}: c=${c}, y=${y}, x=${x}`, data[c + y*224*4 + x*4], tensor.get([0, c, y, x])]);
  var [c,y,x, message] = [1,0,223,"top right"];
  console.log([`${message}: c=${c}, y=${y}, x=${x}`, data[c + y*224*4 + x*4], tensor.get([0, c, y, x])]);
  var [c,y,x, message] = [1,223,223,"bottom right"];
  console.log([`${message}: c=${c}, y=${y}, x=${x}`, data[c + y*224*4 + x*4], tensor.get([0, c, y, x])]);
  return tensor;
}

function getTensorFromArray(data) {
  var width = 224;
  var height = 224;
  var float32Array = new Float32Array(width * height * 3);
  float32Array.fill(500);
  var i = 0;
  const chanSub = [0.485, 0.456, 0.406];
  const chanDiv = [0.229, 0.224, 0.225];
  for (let c = 0; c < 3; c++) {
    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
        // float32Array[i++] = (data[c + (223-y)*224*4 + x*4] / 255.0 - chanSub[c]) / chanDiv[c];
        float32Array[i++] = (1 - chanSub[c]) / chanDiv[c];
      }
    }
  }
  const tensor = new Tensor(float32Array, 'float32', [1, 3, width, height]);
  // is this tensor shaped correctly?
  var [c,y,x, message] = [0,0,223,"top left"];
  console.log([`${message}: c=${c}, y=${y}, x=${x}`, data[c + (223-y)*224*4 + x*4], tensor.get([0, c, x, y])]);
  var [c,y,x, message] = [0,0,0,"bottom left"];
  console.log([`${message}: c=${c}, y=${y}, x=${x}`, data[c + (223-y)*224*4 + x*4], tensor.get([0, c, x, y])]);
  var [c,y,x, message] = [0,223,223,"top right"];
  console.log([`${message}: c=${c}, y=${y}, x=${x}`, data[c + (223-y)*224*4 + x*4], tensor.get([0, c, x, y])]);
  var [c,y,x, message] = [0,223,0,"bottom right"];
  console.log([`${message}: c=${c}, y=${y}, x=${x}`, data[c + (223-y)*224*4 + x*4], tensor.get([0, c, x, y])]);
  return tensor;
}

function runInference(session, imageTensorFloat32) {
  return session.run([imageTensorFloat32]).then(output => {
    // consume the output
    const outputTensor = output.values().next().value;
    console.log(outputTensor);
    return outputTensor;
  }).catch((error) => {
    console.log("inference failed");
    console.log(error);
  });
}

function handleClassProbs(classProbsTensor) {
  setUpClassProbsList();
  var prob = 0.0;
  var imagenetClass = '';
  for (var i = 0; i < 1000; i++) {
    prob = classProbsTensor.data[i];
    imagenetClass = imagenetClasses[i];
    // console.log([imagenetClass, prob]);
    var node = document.getElementById(imagenetClass[0]);
    node.innerHTML = `${imagenetClass[1]}: ${prob}`;
    // var node = document.createElement("li");
    // var textnode = document.createTextNode(`${imagenetClass}: ${prob}`);
    // node.appendChild(textnode);
    // document.getElementById("myList").appendChild(node);
  }
}

button.onclick = function() {
  drawVideoOnCanvas(video, canvas);
  const imageArrayUint8 = getArrayFromCanvas(canvas);
  console.log(imageArrayUint8);
  const imageTensorFloat32 = getTensorFromArray(imageArrayUint8);
  console.log(imageTensorFloat32.get([0, 0, 100, 100]));
  const netOutput = runInference(session, imageTensorFloat32);
  // console.log(["netOutput", netOutput]);
  netOutput.then((classProbsTensor) => {
    console.log("pipeline done");
    // console.log(classProbsTensor);
    // console.log(classProbsTensor.data);
    handleClassProbs(classProbsTensor);
  })
  // console.log(netOutput);
  // console.log(`model output tensor: ${netOutput.data}.`);
};
