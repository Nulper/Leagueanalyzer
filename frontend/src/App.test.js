import { render, screen } from '@testing-library/react';
import App from './App';

test('renders main page correctly', () => {
  render(<App />);
  const headerElement = screen.getByText(/League of Legends Match Analyzer/i);
  expect(headerElement).toBeInTheDocument();
});

test('renders analyze button', () => {
  render(<App />);
  const buttonElement = screen.getByText(/Analyze Matches/i);
  expect(buttonElement).toBeInTheDocument();
});
