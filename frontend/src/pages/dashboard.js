// src/pages/Dashboard.jsx
import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';
import SummaryDisplay from '../components/SummaryDisplay';

const Dashboard = () => {
  const [summary, setSummary] = useState('');

  return (
    <div style={styles.container}>
      <h1 style={styles.header}>Dashboard</h1>
      <FileUpload onSummaryReceived={setSummary} />
      <SummaryDisplay summary={summary} />
    </div>
  );
};

const styles = {
  container: {
    padding: '2rem',
    maxWidth: '800px',
    margin: '0 auto',
  },
  header: {
    textAlign: 'center',
    marginBottom: '2rem',
  },
};

export default Dashboard;
