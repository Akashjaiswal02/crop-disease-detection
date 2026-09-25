import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const handlePredict = async () => {
    if (!image) {
      alert("Please select an image first");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      setLoading(true);
      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData
      );
      setResult(response.data);
    } catch (error) {
      alert("Backend is not running. Backend start karo.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Crop Disease Detection</h1>
      <p>Upload crop leaf image and detect disease using AI.</p>

      <div className="card">
        <input type="file" accept="image/*" onChange={handleImageChange} />

        {preview && <img src={preview} alt="preview" className="preview" />}

        <button onClick={handlePredict}>
          {loading ? "Detecting..." : "Detect Disease"}
        </button>
      </div>

      {result && (
        <div className="result">
          <h2>Prediction Result</h2>
          <p><b>File:</b> {result.filename}</p>
          <p><b>Disease:</b> {result.disease}</p>
          <p><b>Confidence:</b> {result.confidence}%</p>
          <p><b>Treatment:</b> {result.treatment}</p>
        </div>
      )}
    </div>
  );
}

export default App;