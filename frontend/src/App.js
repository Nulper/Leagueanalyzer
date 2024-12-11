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
  const [matchData, setMatchData] = useState([]); // 빈 배열로 초기화
  const [selectedMetric, setSelectedMetric] = useState('KDA');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [predictions, setPredictions] = useState({});

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
        console.log(`Analyzing matches for ${playerName}#${playerTag}`);
        
        const response = await axios.post('http://localhost:5000/api/analyze-matches', {
            name: playerName,
            tag: playerTag
        });

        console.log('Response received:', response.data);

        if (response.data.matches && response.data.matches.length > 0) {
            console.log(`Found ${response.data.matches.length} matches`);
            setMatchData(response.data.matches);
            setPredictions(response.data.predictions || {});
        } else {
            console.log('No matches in response data');
            throw new Error('No matches found');
        }
    } catch (err) {
        console.error('Error details:', {
            message: err.message,
            response: err.response?.data,
            status: err.response?.status
        });
        
        setError(
            err.response?.data?.error || 
            err.message || 
            'Failed to fetch match data'
        );
        setMatchData([]);
        setPredictions({});
    } finally {
        setLoading(false);
    }
};

  const prepareChartData = () => {
    const matchDataPoints = matchData
        .map((match, index) => ({
            gameIndex: index,
            gameCreation: new Date(match.game_creation).toLocaleDateString(),
            [selectedMetric]: match[selectedMetric]
        }))
        .reverse();  // 데이터 순서 뒤집기

    // 예측 데이터 추가
    if (predictions[selectedMetric]) {
        const lastIndex = matchDataPoints.length - 1;
        predictions[selectedMetric].predictions.forEach((pred, index) => {
            matchDataPoints.push({
                gameIndex: lastIndex + 1 + index,
                gameCreation: `Prediction ${index + 1}`,
                [`${selectedMetric}_predicted`]: pred,
                [`${selectedMetric}_lower`]: predictions[selectedMetric].confidence_lower[index],
                [`${selectedMetric}_upper`]: predictions[selectedMetric].confidence_upper[index]
            });
        });
    }

    return matchDataPoints;
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

        {matchData && matchData.length > 0 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Paper elevation={3} style={{ padding: 20 }}>
                <Typography variant="h6">Metric Analysis with Predictions</Typography>
                <select 
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                  style={{ marginBottom: 20, width: '100%', padding: '8px' }}
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
                    name="Actual Data"
                  />
                  
                  {predictions[selectedMetric] && (
                    <>
                      <Line 
                        type="monotone" 
                        dataKey={`${selectedMetric}_predicted`}
                        stroke="#82ca9d"
                        strokeDasharray="5 5"
                        name="Prediction"
                      />
                      <Line 
                        type="monotone" 
                        dataKey={`${selectedMetric}_upper`}
                        stroke="#82ca9d"
                        strokeDasharray="3 3"
                        strokeOpacity={0.3}
                        name="Upper Confidence"
                      />
                      <Line 
                        type="monotone" 
                        dataKey={`${selectedMetric}_lower`}
                        stroke="#82ca9d"
                        strokeDasharray="3 3"
                        strokeOpacity={0.3}
                        name="Lower Confidence"
                      />
                    </>
                  )}
                </LineChart>
              </Paper>
            </Grid>
          </Grid>
        )}
      </Box>
    </Container>
  );
}

export default App;