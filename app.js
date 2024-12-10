import React, { useState } from 'react';
import axios from 'axios';
import { 
  Container, 
  TextField, 
  Button, 
  Typography, 
  Box, 
  Grid, 
  Paper 
} from '@mui/material';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend 
} from 'recharts';

function App() {
  const [playerName, setPlayerName] = useState('');
  const [playerTag, setPlayerTag] = useState('');
  const [matchData, setMatchData] = useState([]);
  const [plotImages, setPlotImages] = useState([]);
  const [selectedMetric, setSelectedMetric] = useState('KDA');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const metrics = [
    'KDA', 'kills', 'deaths', 'assists', 
    'damage_dealt', 'damage_taken', 
    'gold_earned', 'creep_score',
    'damage_per_minute', 'gold_per_minute'
  ];

  const analyzeMatches = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:5000/api/analyze-matches', {
        name: playerName,
        tag: playerTag
      });

      setMatchData(response.data);

      // Fetch plot images
      const plotMetrics = [
        'KDA', 'damage_dealt', 'damage_taken', 
        'gold_earned', 'wards_placed', 
        'creep_score', 'damage_per_minute', 
        'gold_per_minute'
      ];
      
      const plots = plotMetrics.map(metric => 
        `http://localhost:5000/plots/${metric}_trend.png`
      );
      
      setPlotImages(plots);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const prepareChartData = () => {
    return matchData.map(match => ({
      gameCreation: new Date(match.game_creation).toLocaleDateString(),
      [selectedMetric]: match[selectedMetric]
    }));
  };

  return (
    <Container maxWidth="lg">
      <Box my={4}>
        <Typography variant="h4" align="center" gutterBottom>
          League of Legends Match Analyzer
        </Typography>

        <Box display="flex" justifyContent="center" mb={3}>
          <TextField
            label="Summoner Name"
            variant="outlined"
            value={playerName}
            onChange={(e) => setPlayerName(e.target.value)}
            style={{ marginRight: 10 }}
          />
          <TextField
            label="Player Tag"
            variant="outlined"
            value={playerTag}
            onChange={(e) => setPlayerTag(e.target.value)}
          />
          <Button 
            variant="contained" 
            color="primary" 
            onClick={analyzeMatches}
            disabled={loading}
            style={{ marginLeft: 10 }}
          >
            {loading ? 'Analyzing...' : 'Analyze Matches'}
          </Button>
        </Box>

        {error && (
          <Typography color="error" align="center">
            {error}
          </Typography>
        )}

        {matchData.length > 0 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Paper elevation={3} style={{ padding: 20 }}>
                <Typography variant="h6">Metric Analysis</Typography>
                <select 
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                  style={{ marginBottom: 20, width: '100%' }}
                >
                  {metrics.map(metric => (
                    <option key={metric} value={metric}>
                      {metric.toUpperCase()}
                    </option>
                  ))}
                </select>

                <LineChart width={1000} height={400} data={prepareChartData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="gameCreation" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey={selectedMetric} 
                    stroke="#8884d8" 
                    activeDot={{r: 8}} 
                  />
                </LineChart>
              </Paper>
            </Grid>

            {/* Plot Images Section */}
            <Grid item xs={12}>
              <Typography variant="h6" style={{ marginTop: 20 }}>
                Detailed Metric Trends
              </Typography>
              <Grid container spacing={2}>
                {plotImages.map((plotSrc, index) => (
                  <Grid item xs={6} key={index}>
                    <img 
                      src={plotSrc} 
                      alt={`Plot ${index + 1}`} 
                      style={{ width: '100%', maxHeight: 400, objectFit: 'contain' }} 
                    />
                  </Grid>
                ))}
              </Grid>
            </Grid>
          </Grid>
        )}
      </Box>
    </Container>
  );
}

export default App;
