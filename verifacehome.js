model = await tf.loadLayersModel('tfjs_model/model.json');
let model;

async function loadModel() {
    model = await tf.loadLayersModel('tfjs_model/model.json');
    console.log('Model loaded successfully');
}

async function detectFake(img) {
    const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .expandDims();
    const normalized = tensor.div(255.0);
    const prediction = await model.predict(normalized).data();
    return prediction[0];
}

document.getElementById('imageUpload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    const img = new Image();
    img.onload = async () => {
        const score = await detectFake(img);
        const result = score > 0.5 ? 'Fake' : 'Real';
        document.getElementById('result').innerText = `Result: ${result} (Score: ${score.toFixed(2)})`;
    };
    img.src = URL.createObjectURL(file);
});

loadModel();
