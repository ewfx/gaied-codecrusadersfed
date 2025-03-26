import React, { useState } from 'react';
import axios from 'axios';
import FileUpload from './components/FileUpload';
import ResultsDisplay from './components/ResultsDisplay';
import './App.css';

function App() {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState([]);

  const handleUploadSuccess = (files) => {
    setUploadedFiles(files);
  };

  const handleProcess = async () => {
    if (uploadedFiles.length === 0) {
      alert('Please upload files first');
      return;
    }

    setProcessing(true);
    try {
      const response = await axios.post('http://localhost:5000/process');
      setResults(response.data.results);
    } catch (error) {
      console.error('Processing error:', error);
      alert('Error processing files');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>EML File Processor</h1>
      </header>
      <main className="app-main">
        <FileUpload 
          onUploadSuccess={handleUploadSuccess}
          onProcess={handleProcess}
        />
        <ResultsDisplay 
          processing={processing}
          results={results}
        />
      </main>
      <footer className="app-footer">
        <p>EML Processing System © {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

export default App;