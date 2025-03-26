import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ResultsDisplay = ({ processing, results }) => {
  const [expandedFile, setExpandedFile] = useState(null);

  const toggleExpand = (filename) => {
    setExpandedFile(expandedFile === filename ? null : filename);
  };

  if (processing) {
    return <div className="processing-message">Processing files...</div>;
  }

  if (!results || results.length === 0) {
    return <div className="no-results">No results to display</div>;
  }

  return (
    <div className="results-container">
      <h2>Processing Results</h2>
      <div className="results-list">
        {results.map((item, index) => (
          <div key={index} className="result-item">
            <div 
              className="result-header"
              onClick={() => toggleExpand(item.filename)}
            >
              <h3>{item.filename}</h3>
              <span className="toggle-icon">
                {expandedFile === item.filename ? '▼' : '►'}
              </span>
            </div>
            {expandedFile === item.filename && (
              <div className="result-details">
                <div className="result-section">
                  <h4>Metadata</h4>
                  <pre>{JSON.stringify(item.result.metadata, null, 2)}</pre>
                </div>
                <div className="result-section">
                  <h4>Classification</h4>
                  <pre>{JSON.stringify(item.result.classification, null, 2)}</pre>
                </div>
                <div className="result-section">
                  <h4>Extracted Data</h4>
                  <pre>{JSON.stringify(item.result.extracted_data, null, 2)}</pre>
                </div>
                <div className="result-section">
                  <h4>Routing</h4>
                  <pre>{JSON.stringify(item.result.routing, null, 2)}</pre>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ResultsDisplay;