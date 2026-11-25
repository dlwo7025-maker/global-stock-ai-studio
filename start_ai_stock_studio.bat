@echo off
chcp 65001 >nul
title 주린이 AI 모의투자 & 폴리마켓 자동 실행기
cd /d "C:\Users\dlwo7\.gemini\antigravity\scratch\global-stock-predictor"

echo ==============================================================
echo  🌱 주린이 AI 자동 모의투자 + 클라우드 동기화 + 웹 상황판 시작
echo ==============================================================
echo.
echo [클라우드 동기화] 컴퓨터 꺼져 있던 동안 GitHub 서버에 쌓인 최신 데이터 내려받는 중...
git pull --rebase -X theirs
echo.
echo [0/3] 혹시 남은 공백 주가/만기채점 자동 복구 체크 중...
".venv\Scripts\python.exe" catchup_missed_days.py

echo [1/3] 30초 실시간 주가 & 안전브레이크 감시 워커 시작...
start "AI_Live_Paper_Worker" /min ".venv\Scripts\python.exe" live_paper_worker.py --interval 30 --auto-guard

echo [2/3] 3분 주기 AI 승률 자가진화 & 10만원 패키지 자동저장 워커 시작...
start "AI_WinRate_Evolution_Worker" /min ".venv\Scripts\python.exe" evolve_winrate_worker.py --interval 180 --generations 5

echo [3/3] 웹 대시보드 서버 시작 (http://localhost:8501)...
start "AI_Stock_Web_Dashboard" ".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

echo.
echo ✅ 모든 복구 및 백그라운드 프로그램이 실행되었습니다!
echo 👉 잠시 후 인터넷 창에서 http://localhost:8501 화면이 열립니다.
pause
