# Global Stock AI Predictor v2 (국내·해외 주식 통합 AI 예측 플랫폼)

주가 차트뿐 아니라 **다중 종목 통합(Cross-Sectional) 알파 학습 + 회사 재정·실적 5대 기둥 + 실시간 다중소스 이슈·뉴스 이벤트 감성 + 5인 AI 투자 위원회 + 자가 채점(Self-Learning) DB**를 하나로 통합한 국내(KRX) · 해외(US/Global) 주식 예측 시스템입니다.

---

## 핵심 업그레이드 아키텍처 (v2)

1. **다중 종목 통합(Cross-Sectional) 알파 엔진 (`src/stock_predictor/models/cross_sectional.py`)**
   - 국내 대표 20종목 / 해외 대표 20종목의 과거 데이터를 한꺼번에 묶어 **~2만+ 행 패널 데이터셋**으로 통합 학습합니다.
   - 일별 Cross-Sectional 백분위 순위 피처 + Purged 자동 피처 선별(Top 35) + 시장 대비 초과수익(Alpha) 타겟 학습.
   - Top-5 알파 포트폴리오 vs 시장 평균의 거래비용 반영 OOF 벤치마크를 제공합니다.

2. **심층 기업 재정 · 실적 5대 기둥 분석 (`src/stock_predictor/data/fundamentals.py`)**
   - **밸류에이션**(PER/PBR/EV-EBITDA/PEG) · **수익성**(ROE/영업이익률/순이익률) · **성장성**(매출/이익/QoQ) · **재무건전성**(부채비율/유동비율/FCF수익률/순현금) · **실적 서프라이즈 & 컨센서스**(최근 4분기 EPS 서프라이즈, 목표가 괴리율)을 0~100점으로 정량화해 예측 확률과 목표가에 투명하게 반영합니다.
   - `yfinance` + 국내 `네이버 금융` 기업실적 + (선택) `OpenDART` 지원.

3. **실시간 이슈 · 뉴스 이벤트 감성 엔진 (`src/stock_predictor/data/news_intelligence.py`)**
   - 국내(네이버 금융 종목뉴스 + Google News KR + Yahoo Finance) / 해외(Yahoo Finance + Google News US) 다중 수집.
   - **실적/어닝(×1.6)**, **수주/신제품/자사주(×1.4)**, **유상증자/규제/소송(×1.7)**, **증권사 목표가(×1.25)** 등 이벤트별 임팩트 가중치 + 시간 감쇠(Time Decay) 적용. (`OPENAI_API_KEY` + `--use-llm` 시 LLM 정밀 이슈 분석 보강)

4. **자가 채점 & 온라인 가중치 보정 (`src/stock_predictor/storage/memory_db.py`)**
   - 모든 예측을 `data/prediction_memory.db` (SQLite)에 자동 저장하고, 만기 거래일이 지나면 실제 종가로 자동 채점해 최근 실전에서 잘 맞히는 모델 비중을 높입니다.

---

## 빠른 실행

```powershell
cd C:\Users\dlwo7\.gemini\antigravity\scratch\global-stock-predictor
.\.venv\Scripts\Activate.ps1

# 1) 국내(KR) · 해외(US) 20종목 통합 Cross-Sectional 알파 모델 학습 (최초 1회 또는 주간 갱신)
python predict.py --train-universe KR US --horizon 5

# 2) 단일 / 다중 종목 심층 예측 (차트 + 통합알파 + 회사 재무 + 실시간 이슈)
python predict.py 삼성전자 NVDA --horizon 5 --save-json

# 3) 국내 + 해외 관심 종목 일괄 스크리닝
python predict.py 삼성전자 SK하이닉스 현대차 NAVER NVDA AAPL MSFT TSLA --horizon 5 --save-csv

# 4) 만기 지난 과거 예측 자동 채점 및 자가학습 성적표 조회
python predict.py --evaluate

# 5) 인터랙티브 웹 대시보드 실행 (재무 5대 기둥 · 이슈 레이더 · 통합학습 · 자가채점 성적표)
streamlit run app.py
```

> ⚠️ 본 프로젝트의 예측 및 백테스팅 결과는 연구·학습 및 투자 참고용이며, 실제 투자 판단과 손실에 대한 책임은 투자자 본인에게 있습니다.
