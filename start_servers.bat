@echo off
echo Starting League of Legends Match Analyzer...

:: Start the backend server
start cmd /k "cd backend && venv\Scripts\activate && python main.py"

:: Wait for 3 seconds to let backend initialize
timeout /t 3 /nobreak

:: Start the frontend server using the new script
start cmd /k "cd frontend && npm run serve-build"

echo Both servers are starting...
echo Frontend: http://localhost:3000
echo Backend: http://localhost:5000
