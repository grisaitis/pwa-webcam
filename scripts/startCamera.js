var video = document.getElementById("webcam");

var constraints = {
  video: true,
  audio: false
};

function handleSuccess(stream) {
  video.srcObject = stream;
}

function handleError(error) {
  console.log('navigator.getUserMedia error: ', error);
}

navigator.mediaDevices.getUserMedia(constraints)
.then(handleSuccess)
.catch(handleError);
