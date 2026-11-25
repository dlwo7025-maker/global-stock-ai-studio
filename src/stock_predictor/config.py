"""프로젝트 전역 설정, 국내/해외 대표 종목 프리셋, 통합 학습 유니버스, 거래 비용 및 모델 파라미터."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

# 프로젝트 루트 및 캐시/결과/모델/DB 경로
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "saved_models"
DATA_DIR = PROJECT_ROOT / "data"
MEMORY_DB_PATH = DATA_DIR / "prediction_memory.db"

# matplotlib 캐시 권한 경고 방지 (프로젝트 내부 .cache 활용)
if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = CACHE_DIR / "matplotlib"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

# 대표 국내 종목 프리셋 (한글명 -> 6자리 종목코드)
KR_STOCK_PRESETS: dict[str, str] = {
    "삼성전자": "005930",
    "SK하이닉스": "000660",
    "LG에너지솔루션": "373220",
    "삼성바이오로직스": "207940",
    "현대차": "005380",
    "기아": "000270",
    "셀트리온": "068270",
    "KB금융": "105560",
    "신한지주": "055550",
    "하나금융지주": "086790",
    "NAVER": "035420",
    "네이버": "035420",
    "카카오": "035720",
    "POSCO홀딩스": "005490",
    "LG화학": "051910",
    "삼성SDI": "006400",
    "삼성물산": "028260",
    "한화에어로스페이스": "012450",
    "현대로템": "064350",
    "LIG넥스원": "079550",
    "두산에너빌리티": "034020",
    "HD현대일렉트릭": "267260",
    "LS ELECTRIC": "010120",
    "효성중공업": "298040",
    "HD현대중공업": "329180",
    "한화오션": "042660",
    "삼성중공업": "010140",
    "한미반도체": "042700",
    "이수페타시스": "007660",
    "리노공업": "058470",
    "주성엔지니어링": "036930",
    "레인보우로보틱스": "277810",
    "두산로보틱스": "454910",
    "크래프톤": "259960",
    "알테오젠": "196170",
    "리가켐바이오": "141080",
    "에코프로비엠": "247540",
    "에코프로": "086520",
    "KODEX 200": "069500",
}

# Cross-Sectional 통합 학습용 국내 대표 유니버스 (업종 분산 20종목)
KR_UNIVERSE_CODES: list[str] = [
    "005930",  # 삼성전자 (반도체)
    "000660",  # SK하이닉스 (반도체)
    "373220",  # LG에너지솔루션 (2차전지)
    "207940",  # 삼성바이오로직스 (바이오)
    "005380",  # 현대차 (자동차)
    "000270",  # 기아 (자동차)
    "068270",  # 셀트리온 (바이오)
    "105560",  # KB금융 (금융)
    "055550",  # 신한지주 (금융)
    "035420",  # NAVER (인터넷)
    "035720",  # 카카오 (인터넷)
    "005490",  # POSCO홀딩스 (철강/소재)
    "051910",  # LG화학 (화학)
    "006400",  # 삼성SDI (배터리)
    "028260",  # 삼성물산 (지주/건설)
    "012450",  # 한화에어로스페이스 (방산)
    "034020",  # 두산에너빌리티 (원전/에너지)
    "329180",  # HD현대중공업 (조선)
    "259960",  # 크래프톤 (게임)
    "196170",  # 알테오젠 (코스닥 바이오)
]

# 코스닥 대표 종목 코드 (yfinance .KQ 접미사 매핑용)
KNOWN_KOSDAQ_CODES: set[str] = {
    "196170",  # 알테오젠
    "141080",  # 리가켐바이오
    "247540",  # 에코프로비엠
    "086520",  # 에코프로
    "028300",  # HLB
    "058470",  # 리노공업
    "036930",  # 주성엔지니어링
    "240810",  # 원익IPS
    "277810",  # 레인보우로보틱스
    "035900",  # JYP Ent.
    "041510",  # 에스엠
    "263750",  # 펄어비스
    "293490",  # 카카오게임즈
    "145020",  # 휴젤
    "214150",  # 클래시스
}

# 대표 해외(미국) 종목 프리셋 (표시명 -> 티커)
US_STOCK_PRESETS: dict[str, str] = {
    "NVIDIA (엔비디아)": "NVDA",
    "Micron (마이크론)": "MU",
    "Apple (애플)": "AAPL",
    "Microsoft (마이크로소프트)": "MSFT",
    "Alphabet (구글)": "GOOGL",
    "Amazon (아마존)": "AMZN",
    "Meta (메타)": "META",
    "Tesla (테슬라)": "TSLA",
    "TSMC (대만반도체)": "TSM",
    "Broadcom (브로드컴)": "AVGO",
    "AMD": "AMD",
    "Palantir (팔란티어)": "PLTR",
    "CoreWeave (코어위브)": "CRWV",
    "Oklo (오클로 원전)": "OKLO",
    "Constellation Energy": "CEG",
    "IonQ (아이온큐 양자)": "IONQ",
    "D-Wave Quantum": "QBTS",
    "Moderna (모더나)": "MRNA",
    "Netflix (넷플릭스)": "NFLX",
    "Eli Lilly (일라이릴리)": "LLY",
    "HCA Healthcare": "HCA",
    "Global Payments": "GPN",
    "Walmart (월마트)": "WMT",
    "JPMorgan Chase": "JPM",
    "Visa": "V",
    "Costco (코스트코)": "COST",
    "Exxon Mobil": "XOM",
    "UnitedHealth": "UNH",
    "Berkshire Hathaway": "BRK-B",
    "S&P 500 ETF": "SPY",
    "Nasdaq 100 ETF": "QQQ",
    "SOXX (반도체 ETF)": "SOXX",
}

# Cross-Sectional 통합 학습용 미국 대표 유니버스 (섹터 분산 20종목)
US_UNIVERSE_TICKERS: list[str] = [
    "NVDA",   # 반도체/AI
    "AAPL",   # 빅테크
    "MSFT",   # 빅테크/클라우드
    "GOOGL",  # 빅테크/AI
    "AMZN",   # 빅테크/커머스
    "META",   # 빅테크/광고
    "TSLA",   # 전기차/로보틱스
    "AVGO",   # 반도체
    "AMD",    # 반도체
    "TSM",    # 파운드리
    "NFLX",   # 미디어
    "PLTR",   # AI 소프트웨어
    "LLY",    # 헬스케어/제약
    "UNH",    # 헬스케어
    "JPM",    # 금융
    "V",      # 결제/금융
    "COST",   # 소비재
    "XOM",    # 에너지
    "CAT",    # 산업재
    "BRK-B",  # 복합/가치
]

# 글로벌 거시경제 & 벤치마크 티커 (yfinance)
MACRO_TICKERS: dict[str, str] = {
    "sp500": "^GSPC",       # 미국 S&P 500
    "nasdaq": "^IXIC",      # 미국 나스닥 종합
    "kospi": "^KS11",       # 한국 코스피 지수
    "vix": "^VIX",          # 변동성(공포) 지수
    "us10y": "^TNX",        # 미국 10년물 국채 금리
    "usdkrw": "KRW=X",      # 원/달러 환율
    "dxy": "DX-Y.NYB",      # 달러 인덱스
}

# 시장별 현실적 거래 비용(편도 기준: 수수료 + 슬리피지 + 세금 평균 반영)
MARKET_COST_BPS: dict[str, dict[str, float]] = {
    "KR": {
        "commission": 0.00015,   # 온라인 증권사 수수료 약 0.015%
        "sell_tax": 0.0018,      # 코스피/코스닥 매도 증권거래세 약 0.18%
        "slippage": 0.0005,      # 슬리피지 0.05%
    },
    "US": {
        "commission": 0.0005,    # 해외주식 수수료 약 0.05%
        "sell_tax": 0.00003,     # SEC Fee 등 소액
        "slippage": 0.0005,      # 슬리피지 0.05%
    },
}


@dataclass
class PredictorConfig:
    """예측·백테스팅 파이프라인 설정."""

    horizon: int = 5                     # 예측 기간 (거래일: 1, 5, 10, 20 등)
    lookback_years: int = 5              # 학습에 사용할 과거 데이터 연수
    n_splits: int = 5                    # Walk-Forward 검증 분할 수
    min_train_rows: int = 240            # 최소 학습 거래일 수 (약 1년)
    use_deep_learning: bool = True       # 시계열 딥러닝 모델 포함 여부
    use_chronos: bool = False            # Amazon Chronos 파운데이션 모델 포함 여부 (선택)
    use_cross_sectional: bool = True     # 학습된 다중종목 통합(Cross-Sectional) 알파 모델 결합 여부
    use_fundamental_overlay: bool = True # 기업 재정·실적 퀄리티 예측 반영 여부
    use_news_overlay: bool = True        # 실시간 이슈·뉴스 감성 충격 예측 반영 여부
    use_alt_overlay: bool = True         # 폴리마켓 예측시장·레딧 민심·임원/외인기관 큰손 수급 예측 반영 여부
    use_memory_calibration: bool = True  # SQLite 실전 채점 기록 기반 모델 가중치 보정 여부
    record_predictions: bool = True      # 예측 결과를 자가학습 SQLite DB에 자동 기록할지 여부
    use_llm_agents: bool = False         # OpenAI API 기반 LLM 이슈/CIO 분석 활성화 여부
    use_cache: bool = True               # 당일 수집 데이터 로컬 캐시 사용 여부
    cache_ttl_hours: int = 4             # 캐시 유효 시간
    long_threshold: float = 0.005        # 매수 시그널 기준 기대수익률 (+0.5%)
    prob_threshold: float = 0.53         # 매수 시그널 기준 상승확률 (53%)
    allow_short: bool = False            # 공매도 허용 여부 (기본: 롱-현금 전략)
    top_k_features: int = 35             # Cross-Sectional 자동 피처 선별 상위 개수
    random_seed: int = 42
    custom_macro_tickers: dict[str, str] = field(default_factory=lambda: dict(MACRO_TICKERS))
