import('./runBrowserTests.js');
import('./startCamera.js');
import('./setUpButton.js').then(() => {console.log("dooonnnne");});
import('./runOnnxTest.js').then(() => {console.log("ran onnx.js test");});
