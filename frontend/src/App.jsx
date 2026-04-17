import React, { useState, useRef } from 'react';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type.startsWith('audio/')) {
      setFile(selectedFile);
      setError(null);
      setResult(null);
    } else {
      setError('Please select a valid audio file.');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = () => {
    setDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('audio/')) {
      setFile(droppedFile);
      setError(null);
      setResult(null);
    } else {
      setError('Please drop a valid audio file.');
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/extract-vocals`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.response?.data?.detail || 'An error occurred during processing. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="container">
      <header>
        <h1>Vocal Extractor</h1>
        <p className="subtitle">High-quality vocal extraction powered by AI</p>
      </header>

      <main className="app-card">
        {!loading && !result ? (
          <div className="upload-section">
            <div 
              className={`upload-zone ${dragging ? 'dragging' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <svg className="upload-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
              </svg>
              <p className="upload-text">{file ? file.name : 'Click or drop audio file here'}</p>
              <p className="upload-hint">Supports MP3, WAV, M4A up to 50MB</p>
              <input 
                type="file" 
                className="hidden-input" 
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="audio/*"
              />
            </div>

            {error && <p className="error-message" style={{ color: 'var(--error)', marginTop: '1rem', textAlign: 'center' }}>{error}</p>}

            <div style={{ marginTop: '2rem', textAlign: 'center' }}>
              <button 
                className="btn btn-primary" 
                disabled={!file}
                onClick={handleUpload}
              >
                Extract Vocals
              </button>
            </div>
          </div>
        ) : loading ? (
          <div className="status-view">
            <div className="loader"></div>
            <p className="upload-text">Extracting vocals...</p>
            <p className="upload-hint">This usually takes 10-30 seconds depending on file size</p>
          </div>
        ) : (
          <div className="result-view">
            <div className="result-header">
              <div className="file-info">
                <span className="success-badge">
                  <svg width="20" height="20" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"></path>
                  </svg>
                  Extraction Complete
                </span>
              </div>
              <button className="btn" onClick={reset} style={{ background: 'transparent', color: 'var(--text-muted)' }}>Process another</button>
            </div>

            <div className="audio-player-container">
              <p style={{ marginBottom: '1rem', fontWeight: 600 }}>Preview Extracted Vocals:</p>
              <audio controls src={`${API_BASE_URL}${result.result_url}`}>
                Your browser does not support the audio element.
              </audio>
            </div>

            <div style={{ textAlign: 'center' }}>
              <a 
                href={`${API_BASE_URL}/download/${result.result_url.split('/').pop()}`} 
                className="btn btn-primary"
                style={{ textDecoration: 'none' }}
              >
                Download Vocals
              </a>
            </div>
          </div>
        )}
      </main>

      <footer style={{ marginTop: 'auto', textAlign: 'center', padding: '2rem 0', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
        &copy; 2026 Vocal Extractor AI. All rights reserved.
      </footer>
    </div>
  );
}

export default App;
