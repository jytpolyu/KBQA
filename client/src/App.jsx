import React, { useState } from 'react';
import { Radio, Select, Input, Button, Typography, Space, message, Divider } from 'antd';
import axios from 'axios';
import './App.css';

const { Title, Text } = Typography;
const { Option } = Select;

const SearchPage = () => {
  const [mode, setMode] = useState('search');
  const [algorithm, setAlgorithm] = useState('TF-IDF');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);

  const handleSearch = async () => {
    if (!query) {
      message.warning('Please enter a query');
      return;
    }

    setLoading(true);
    try {
      // 如果是answer模式，则固定method为FAISS
      const method = mode === 'answer' ? 'FAISS' : algorithm;
      
      const response = await axios.post(
        'http://127.0.0.1:8000/api/search',
        { query, mode, method },
        { headers: { 'Content-Type': 'application/json' } }
      );

      console.log('Response:', response.data);

      if (mode === 'search') {
        if (response.data.results && !Array.isArray(response.data.results)) {
          message.error('Invalid result format');
          setResults([]);
          return;
        }
        setResults(response.data.results || []);
      } else if (mode === 'answer') {
        setResults(response.data.results || 'No answer generated');
      }

    } catch (err) {
      console.error(err);
      message.error('Request failed. Please check if the backend service is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Title level={2} className="title">📚 Document Retrieval & Question Answering</Title>
  
      <div className="main-content">
        {/* Left Panel */}
        <div className="search-panel">
          <div className="search-card">
            <div className="section">
              <Text strong style={{ display: 'inline-block', width: '100px' }}>Select Mode:</Text>
              <Radio.Group value={mode} onChange={(e) => setMode(e.target.value)}>
                <Radio value="search">Document Retrieval</Radio>
                <Radio value="answer">Question Answering</Radio>
              </Radio.Group>
            </div>
  
            <Divider />
  
            <div className="section">
              {/* 只在search模式下显示Algorithm选择模块 */}
              {mode === 'search' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: 12 }}>
                  <Text strong style={{ whiteSpace: 'nowrap', width: '80px' }}>Algorithm:</Text>
                  <Select
                    width={120}
                    value={algorithm}
                    onChange={setAlgorithm}
                    style={{ width: '100%' }}
                  >
                    <Option value="TF-IDF">TF-IDF</Option>
                    <Option value="BM25">BM25</Option>
                    <Option value="FAISS">FAISS</Option>
                    <Option value="GloVe">GloVe</Option>
                  </Select>
                </div>
              )}

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: 12 }}>
                <Text strong style={{ whiteSpace: 'nowrap', width: '80px'  }}>Query:</Text>
                <Input
                  placeholder="Enter your query"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  allowClear
                />
              </div>
  
              <Button type="primary" block style={{ marginTop: 16 }} loading={loading} onClick={handleSearch}>
                {mode === 'search' ? 'Search' : 'Generate Answer'}
              </Button>
            </div>
          </div>
        </div>
  
        {/* Right Panel */}
        <div className="results-panel">
          {
            mode === 'search' && Array.isArray(results) && results.length > 0 && (
              <div className="results-card">
                <Title level={4}>🔎 Search Results ({results.length})</Title>
                <div className="results-list">
                  {results.map((item, index) => (
                    <div className="result-item" key={index}>
                      <Text strong>ID:</Text> {item.document_id}<br />
                      <div className='result-text'>
                        <Text>Content:</Text> {item.document_text}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )
          }
          {
            mode === 'answer' && typeof results === 'string' && results && (
              <div className="results-card">
                <Title level={4}>💡 Answer</Title>
                <div className="answer-box">
                  <Text>{results}</Text>
                </div>
              </div>
            )
          }
        </div>
      </div>
    </div>
  );
};

export default SearchPage;
