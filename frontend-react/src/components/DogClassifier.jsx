import React, { useState, useRef } from "react";
import { apiClient } from "../services/api-client";
import "./DogClassifier.css";

const DogClassifier = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleClassify = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.classifyImage(selectedImage);
      setResult(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="dog-classifier-container">
      <h3>🐶 Identificador de Razas de Perros</h3>

      <div className="upload-section">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          ref={fileInputRef}
          className="file-input"
          id="dog-image-upload"
        />
        <label htmlFor="dog-image-upload" className="btn btn-secondary">
          Seleccionar Imagen
        </label>
      </div>

      {previewUrl && (
        <div className="preview-section">
          <img src={previewUrl} alt="Preview" className="image-preview" />

          <div className="action-buttons">
            <button
              onClick={handleClassify}
              disabled={loading}
              className="btn btn-primary"
            >
              {loading ? "Analizando..." : "Identificar Raza"}
            </button>
            <button
              onClick={handleClear}
              className="btn btn-outline"
              disabled={loading}
            >
              Limpiar
            </button>
          </div>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}

      {result && (
        <div className="result-card">
          <h4>Resultado:</h4>
          <div className="prediction-info">
            <span className="breed-name">
              {result.prediction.replace(/_/g, " ").toUpperCase()}
            </span>
            {result.confidence && (
              <span className="confidence-badge">
                {(result.confidence * 100).toFixed(1)}% Confianza
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default DogClassifier;
