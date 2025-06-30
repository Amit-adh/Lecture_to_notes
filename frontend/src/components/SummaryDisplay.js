import React from 'react';

const SummaryDisplay = ({ summary }) => {
  return (
    <div>
      <h3>Summarized Notes</h3>
      <div style={{
        border: '1px solid #ccc',
        padding: '1rem',
        minHeight: '150px',
        backgroundColor: '#f9f9f9'
      }}>
        {summary ? summary : 'No summary to display yet.'}
      </div>
    </div>
  );
};

export default SummaryDisplay;
