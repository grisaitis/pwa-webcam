const onnx = window.onnx;


function getInputsAdd() {
  const x = new Float32Array(3 * 4 * 5).fill(1);
  const y = new Float32Array(3 * 4 * 5).fill(2);
  const tensorX = new onnx.Tensor(x, 'float32', [3, 4, 5]);
  const tensorY = new onnx.Tensor(y, 'float32', [3, 4, 5]);
  return [tensorX, tensorY];
}

function getInputsSqueezeNetV1() {
  const x = new Float32Array(1*3*224*224).fill(0.5);
  const tensorX = new Tensor(x, 'float32', [1,3,224,224]);
  return [tensorX]
}

// load the ONNX model file
// const getInputs = getInputsSqueezeNetV1;
// const modelURL = "https://raw.githubusercontent.com/Microsoft/onnxjs-demo/data/data/examples/models/squeezenetV1_8.onnx";

const getInputs = getInputsAdd;
const modelURL = "https://raw.githubusercontent.com/Microsoft/onnxjs-demo/data/data/examples/models/add.onnx";

// create a onnx.js "session"
const session = new onnx.InferenceSession();

session.loadModel(modelURL)
.then(() => {
  // generate model input
  const inferenceInputs = getInputs();
  // execute the model
  session.run(inferenceInputs)
  .then(output => {
    // consume the output
    const outputTensor = output.values().next().value;
    console.log(`model output tensor: ${outputTensor.data}.`);
  });
})
.catch((e) => {
  console.log(e);
});
