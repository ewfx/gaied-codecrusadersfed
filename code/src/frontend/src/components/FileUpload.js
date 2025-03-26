import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = ({ onUploadSuccess, onProcess }) => {
  const [files, setFiles] = useState([]);
  const [uploadMessage, setUploadMessage] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
    setUploadMessage('');
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setUploadMessage('Please select at least one file');
      return;
    }

    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    setIsUploading(true);
    setUploadMessage('Uploading files...');

    try {
      const response = await axios.post('http://localhost:5000/upload', formData);
      setUploadMessage(response.data.message);
      onUploadSuccess(response.data.files);
    } catch (error) {
      setUploadMessage('Error uploading files');
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="file-upload-container">
      <h2>Upload EML Files</h2>
      <input
        type="file"
        id="eml-upload"
        onChange={handleFileChange}
        multiple
        accept=".eml"
        disabled={isUploading}
      />
      <div className="button-group">
        <button
          onClick={handleUpload}
          disabled={isUploading || files.length === 0}
        >
          {isUploading ? 'Uploading...' : 'Upload Files'}
        </button>
        <button
          onClick={onProcess}
          disabled={isUploading}
          className="process-button"
        >
          Process Files
        </button>
      </div>
      {uploadMessage && <p className="message">{uploadMessage}</p>}
      {files.length > 0 && (
        <div className="file-list">
          <h4>Selected Files:</h4>
          <ul>
            {files.map((file, index) => (
              <li key={index}>{file.name}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default FileUpload;