import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = ({ onSummaryReceived }) => {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a file first");

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Assuming the backend returns JSON: { summary: "..." }
      onSummaryReceived(response.data.summary);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading or processing file.');
    }
  };

  return (
    <div style={{ marginBottom: '2rem' }}>
      <h3>Upload Lecture File</h3>
      <input type="file" accept=".pdf,.ppt,.pptx,.mp3,.mp4" onChange={handleFileChange} />
      <button onClick={handleUpload} style={{ marginLeft: '1rem' }}>
        Upload
      </button>
    </div>
  );
};

export default FileUpload;
