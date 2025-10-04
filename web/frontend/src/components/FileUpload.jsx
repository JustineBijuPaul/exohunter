import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const FileUpload = ({ onFileSelect, loading }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.csv'],
    },
    multiple: false,
    disabled: loading,
  });

  return (
    <div className="card">
      <h2>📁 File Upload</h2>
      
      <div className="instructions">
        <h3>Supported File Formats:</h3>
        <ul>
          <li><strong>Tabular Features:</strong> CSV with numerical feature columns</li>
          <li><strong>Light Curve Data:</strong> CSV with 'time' and 'flux' columns</li>
        </ul>
      </div>

      <div
        {...getRootProps()}
        className={`upload-zone ${isDragActive ? 'dragover' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="file-icon">📄</div>
        {isDragActive ? (
          <p>Drop the CSV file here...</p>
        ) : (
          <>
            <p>Drag & drop a CSV file here, or click to select</p>
            <p style={{ fontSize: '14px', color: '#6c757d' }}>
              Maximum file size: 10MB
            </p>
          </>
        )}
      </div>

      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          Processing file...
        </div>
      )}
    </div>
  );
};

export default FileUpload;
