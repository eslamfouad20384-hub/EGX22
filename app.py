from __future__ import annotations
import math, time, threading, os, logging, platform, sys, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st
import yfinance as yf

# Production: disable yfinance's configurable network retry counter. This does NOT
# mean one HTTP request per yfinance operation: yfinance may still perform internal
# cookie/crumb strategy requests. V21 tracks those separately and never transport-retries
# cookie/crumb requests in the application layer.
try:
    yf.config.network.retries = 0
except Exception:
    try:
        yf.set_config(retries=0)
    except Exception:
        pass

VERSION = 'V22.0 QA/PRODUCTION'

# Production diagnostics/logging
LOG_LEVEL = os.getenv('EGX_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('egx_analyzer')

SYMBOLS = list(dict.fromkeys('COMI MFPC ABUK ETEL PHDC TMGH HRHO FWRY SWDY ORAS EFGH AMOC HELI SODIC EGCH ACRI EMFD ADIB CIEB QNBA FAIT JUFO EAST CLHO CCAP AUTO ESRS ORWE SKPC EKHO RMDA ISPH OCDI MNHD PORT TAQA ATQA ARCC ARAB AIVC BTFH BINV DOMT EFID MCQE MOIL MTIE RACC RAYA SAUD UEGC ZMID MICH NCCW NCC IRON MASR MPRC ODID ELSH OIH ODCO NILE ENGC EGTS ATLC OLFI POUL UNIT VERT COSG DAPH ARPI RREI TALM CIRA EDBM MENA MEPA MPCO NEDA PRCL SCEM SDTI TANM TASG TRTO UNIP UASG ZEOT ZAHM ECAP MILS FERC IFAP MCRO NIPH SPIN KIMA SAIB'.split()))

def _yahoo_request_kind(url):
    u=str(url).lower()
    if 'getcrumb' in u or 'crumb' in u:
        return 'crumb'
    if 'fc.yahoo.com' in u or '/v1/test/getcrumb' in u or 'cookie' in u:
        return 'cookie'
    return 'data'

class GlobalYahooGate:
    """Global pacing + cooldown shared by all worker sessions."""
    def __init__(self, interval=1.0):
        self.interval = max(0.0, float(interval))
        self.lock = threading.Lock()
        self.last_request = 0.0
        self.cooldown_until = 0.0
        self.request_count = 0
        self.retry_count = 0
        self.rate_limit_count = 0
        self.last_cooldown_seconds = 0.0
        self.transport_attempts = 0
        self.cookie_requests = 0
        self.crumb_requests = 0
        self.data_requests = 0
        self.cookie_responses = 0
        self.crumb_responses = 0
        self.yfinance_internal_retry_possible = True
        self.yfinance_singleton_lock_serializes_calls = True

    def wait(self, kind='data'):
        while True:
            with self.lock:
                now = time.monotonic()
                target = max(self.last_request + self.interval, self.cooldown_until)
                delay = target - now
                if delay <= 0:
                    self.last_request = time.monotonic()
                    self.request_count += 1
                    self.transport_attempts += 1
                    if kind == 'crumb': self.crumb_requests += 1
                    elif kind == 'cookie': self.cookie_requests += 1
                    else: self.data_requests += 1
                    return
            time.sleep(min(delay, 5.0))

    def record_response(self, kind, status_code):
        with self.lock:
            if int(status_code) == 429:
                self.rate_limit_count += 1
            if kind == 'crumb': self.crumb_responses += 1
            elif kind == 'cookie': self.cookie_responses += 1

    def cooldown(self, seconds):
        with self.lock:
            seconds=max(0.0,float(seconds))
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)
            self.last_cooldown_seconds=seconds


def _retry_after_seconds(response):
    value = response.headers.get('Retry-After') if response is not None else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except Exception:
        try:
            dt = pd.Timestamp(value, tz='UTC')
            now = pd.Timestamp.now(tz='UTC')
            return max(0.0, (dt - now).total_seconds())
        except Exception:
            return None


class ManagedRequestsSession(requests.Session):
    """Explicit requests fallback with narrow transport retry policy."""
    def __init__(self, gate, retries=2, timeout=20):
        super().__init__()
        self.gate = gate
        self.egx_retries = int(max(0, retries))
        self.egx_timeout = int(timeout)
        self.headers.update({'User-Agent': 'Mozilla/5.0 EGXSmartAnalyzer/21.0'})
        adapter = HTTPAdapter(max_retries=Retry(total=0), pool_connections=10, pool_maxsize=10)
        self.mount('https://', adapter); self.mount('http://', adapter)

    def request(self, method, url, **kwargs):
        kwargs.setdefault('timeout', self.egx_timeout)
        kind=_yahoo_request_kind(url)
        max_attempts=1 if kind in ('cookie','crumb') else self.egx_retries+1
        for attempt in range(max_attempts):
            self.gate.wait(kind)
            try:
                response=super().request(method,url,**kwargs)
                self.gate.record_response(kind,response.status_code)
                if response.status_code==429:
                    delay=_retry_after_seconds(response)
                    self.gate.cooldown(delay if delay is not None else min(60.0,2.0**attempt))
                    if kind not in ('cookie','crumb') and attempt < max_attempts-1:
                        self.gate.retry_count += 1
                        continue
                elif response.status_code in (500,502,503,504) and kind not in ('cookie','crumb') and attempt < max_attempts-1:
                    self.gate.cooldown(min(30.0,2.0**attempt))
                    self.gate.retry_count += 1
                    continue
                return response
            except requests.RequestException:
                if kind in ('cookie','crumb') or attempt >= max_attempts-1:
                    raise
                self.gate.cooldown(min(30.0,2.0**attempt))
                self.gate.retry_count += 1
        raise RuntimeError('Yahoo HTTP request failed')


try:
    from curl_cffi import requests as curl_requests
    CURL_CFFI_AVAILABLE = True
    _CURL_REQUEST_EXCEPTIONS = (curl_requests.exceptions.RequestException,)
except Exception:
    curl_requests = None
    CURL_CFFI_AVAILABLE = False
    _CURL_REQUEST_EXCEPTIONS = (Exception,)


if CURL_CFFI_AVAILABLE:
    class ManagedCurlSession(curl_requests.Session):
        """A REAL curl_cffi Session subclass accepted by yfinance.
        Global pacing/retry is added by overriding request(), without wrapping
        the Session object, so yfinance's session type contract remains intact.
        """
        def __init__(self, gate, retries=2, timeout=20):
            super().__init__(impersonate='chrome', retry=0)
            self.gate=gate
            self.egx_retries=int(max(0,retries))
            self.egx_timeout=int(timeout)
            self.headers.update({'User-Agent':'Mozilla/5.0 EGXSmartAnalyzer/21.0'})
        def request(self, method, url, **kwargs):
            kwargs.setdefault('timeout', self.egx_timeout)
            kind=_yahoo_request_kind(url)
            max_attempts=1 if kind in ('cookie','crumb') else self.egx_retries+1
            last=None
            for attempt in range(max_attempts):
                self.gate.wait(kind)
                try:
                    response=super().request(method,url,**kwargs)
                    self.gate.record_response(kind,response.status_code)
                    if response.status_code==429:
                        delay=_retry_after_seconds(response)
                        self.gate.cooldown(delay if delay is not None else min(120.0,2.0**attempt))
                        if kind not in ('cookie','crumb') and attempt < max_attempts-1:
                            self.gate.retry_count += 1
                            continue
                    elif response.status_code in (500,502,503,504) and kind not in ('cookie','crumb') and attempt < max_attempts-1:
                        self.gate.cooldown(min(60.0,2.0**attempt))
                        self.gate.retry_count += 1
                        continue
                    return response
                except _CURL_REQUEST_EXCEPTIONS as exc:
                    last=exc
                    if kind in ('cookie','crumb') or attempt >= max_attempts-1:
                        raise
                    self.gate.cooldown(min(60.0,2.0**attempt))
                    self.gate.retry_count += 1
            raise last or RuntimeError('Yahoo HTTP request failed')

else:
    ManagedCurlSession = None

class YahooHTTPManager:
    """Production Yahoo manager with one shared Session per manager.

    yfinance internally uses shared YfData/cookie/crumb state. To avoid races
    between worker threads, the manager owns ONE real curl_cffi/requests Session
    and serializes yfinance operations through a lock. The worker pool remains
    parallel at the analysis level, while Yahoo transport is globally paced and
    safe. curl_cffi and yfinance retries are both disabled; only this layer may
    retry transient data requests. Cookie/crumb operations are never retried by
    this layer.
    """
    def __init__(self, interval=1.0, retries=2, timeout=20):
        self.interval=float(interval); self.retries=int(retries); self.timeout=int(timeout)
        self.gate=GlobalYahooGate(interval)
        self._session_obj=None
        self._session_lock=threading.Lock()
        self._yf_lock=threading.RLock()

    def _session(self):
        with self._session_lock:
            if self._session_obj is None:
                if CURL_CFFI_AVAILABLE and os.getenv('EGX_FORCE_REQUESTS','0')!='1':
                    self._session_obj=ManagedCurlSession(self.gate,self.retries,self.timeout)
                else:
                    self._session_obj=ManagedRequestsSession(self.gate,self.retries,self.timeout)
            return self._session_obj

    @property
    def session(self):
        return self._session()

    def call(self, fn):
        # yfinance's YfData/cookie/crumb state is shared internally; serialize
        # each yfinance operation so worker threads cannot mutate it concurrently.
        with self._yf_lock:
            return fn()

    def ticker(self,symbol):
        with self._yf_lock:
            return yf.Ticker(symbol+'.CA',session=self._session())

    def history(self,ticker):
        try:
            x=clean(self.call(lambda: ticker.history(period='5y',interval='1d',auto_adjust=False,actions=True,repair=True,timeout=self.timeout)))
            return x,'success',''
        except TypeError as exc:
            msg=str(exc).lower()
            unsupported=('repair' in msg and ('keyword' in msg or 'unexpected' in msg or 'argument' in msg))
            if not unsupported:
                return pd.DataFrame(),f'history_failed:{type(exc).__name__}',str(exc)[:240]
            try:
                x=clean(self.call(lambda: ticker.history(period='5y',interval='1d',auto_adjust=False,actions=True,timeout=self.timeout)))
                return x,'unsupported_repair_fallback','repair=True unsupported'
            except Exception as exc2:
                return pd.DataFrame(),f'history_fallback_failed:{type(exc2).__name__}',str(exc2)[:240]
        except Exception as exc:
            return pd.DataFrame(),f'history_failed:{type(exc).__name__}',str(exc)[:240]

    def info(self,ticker):
        try:
            x=self.call(lambda: ticker.get_info() if hasattr(ticker,'get_info') else ticker.info)
            if isinstance(x,dict) and x:
                return x,'ok',''
            return {},'empty','Yahoo returned empty info'
        except Exception as exc:
            return {},'error',f'{type(exc).__name__}: {str(exc)[:220]}'

    def income_stmt(self,ticker):
        try:
            x=self.call(lambda: ticker.income_stmt)
            if isinstance(x,pd.DataFrame) and not x.empty:
                return x,'ok',''
            return pd.DataFrame(),'empty','Yahoo returned empty income statement'
        except Exception as exc:
            return pd.DataFrame(),'error',f'{type(exc).__name__}: {str(exc)[:220]}'

    def actions(self,ticker):
        try:
            x=self.call(lambda: ticker.actions)
            if isinstance(x,pd.DataFrame) and not x.empty:
                return x,'ok',''
            return pd.DataFrame(),'empty','Yahoo returned empty actions'
        except Exception as exc:
            return pd.DataFrame(),'error',f'{type(exc).__name__}: {str(exc)[:220]}'

def clean(x):
    if x is None or x.empty: return pd.DataFrame()
    x=x.copy()
    if isinstance(x.columns,pd.MultiIndex): x.columns=[c[-1] for c in x.columns]
    x=x.rename(columns={c:str(c).title() for c in x.columns})
    for c in ['Open','High','Low','Close','Volume','Adj Close','Stock Splits','Dividends']:
        if c in x: x[c]=pd.to_numeric(x[c],errors='coerce')
    for c in ['Open','High','Low','Close','Volume']:
        if c not in x: x[c]=np.nan
    x=x[~x.index.duplicated(keep='last')].sort_index()
    return x.dropna(subset=['Close'])

def ema(s,n): return s.ewm(span=n,adjust=False,min_periods=n).mean()
def rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0); ag=g.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); al=l.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); rs=ag/al.replace(0,np.nan); return (100-100/(1+rs)).where(al.ne(0),100)
def atr(x,n=14):
    p=x.Close.shift(); tr=pd.concat([x.High-x.Low,(x.High-p).abs(),(x.Low-p).abs()],axis=1).max(axis=1); return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def adx(x,n=14):
    up=x.High.diff(); dn=-x.Low.diff(); plus=up.where((up>dn)&(up>0),0); minus=dn.where((dn>up)&(dn>0),0); p=x.Close.shift(); tr=pd.concat([x.High-x.Low,(x.High-p).abs(),(x.Low-p).abs()],axis=1).max(axis=1); a=tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); pi=100*plus.ewm(alpha=1/n,adjust=False,min_periods=n).mean()/a.replace(0,np.nan); mi=100*minus.ewm(alpha=1/n,adjust=False,min_periods=n).mean()/a.replace(0,np.nan); dx=100*(pi-mi).abs()/(pi+mi).replace(0,np.nan); return dx.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def obv(x): return (np.sign(x.Close.diff()).fillna(0)*x.Volume.fillna(0)).cumsum()

def add_ind(x):
    x=clean(x)
    if x.empty:return x
    x['EMA20']=ema(x.Close,20);x['EMA50']=ema(x.Close,50);x['EMA200']=ema(x.Close,200);x['RSI']=rsi(x.Close)
    macd=ema(x.Close,12)-ema(x.Close,26);x['MACD']=macd;x['MACD_SIGNAL']=macd.ewm(span=9,adjust=False,min_periods=9).mean();x['ATR']=atr(x);x['ADX']=adx(x);x['OBV']=obv(x);x['VOL20']=x.Volume.rolling(20,min_periods=20).mean();x['RET20']=x.Close.pct_change(20)
    x['RES20']=x.High.shift(1).rolling(20,min_periods=10).max();x['SUP20']=x.Low.shift(1).rolling(20,min_periods=10).min()
    return x

def completed_tf(d):
    d=clean(d); w=d.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna(subset=['Close']); m=d.resample('ME').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna(subset=['Close']); last=d.index[-1]
    if len(w) and w.index[-1].date()>=last.date(): w=w.iloc[:-1]
    if len(m) and (m.index[-1].year,m.index[-1].month)==(last.year,last.month): m=m.iloc[:-1]
    return add_ind(w),add_ind(m)

def cagr_series(s):
    s=pd.to_numeric(s,errors='coerce').dropna()
    if len(s)<2:return None,'insufficient'
    a,b=float(s.iloc[0]),float(s.iloc[-1]); days=(pd.Timestamp(s.index[-1])-pd.Timestamp(s.index[0])).days
    if a<=0 and b>0:return None,'turnaround'
    if a<=0 or b<=0 or days<=365:return None,'invalid'
    return (b/a)**(365.25/days)-1,'ok'

def split_adjusted_price(d,actions=None):
    out=pd.to_numeric(d.Close,errors='coerce').copy()
    if out.empty or actions is None or actions.empty or 'Stock Splits' not in actions.columns:return out
    splits=pd.to_numeric(actions['Stock Splits'],errors='coerce').dropna();splits=splits[splits>0]
    for raw_dt,ratio in splits.items():
        try:
            dt=pd.Timestamp(raw_dt); idx_tz=getattr(out.index,'tz',None); dt=dt.tz_localize(idx_tz) if idx_tz is not None and dt.tz is None else (dt.tz_convert(idx_tz) if idx_tz is not None else dt.tz_localize(None) if dt.tz is not None else dt)
            out.loc[out.index<dt]*=1/float(ratio)
        except Exception: continue
    return out

def price_cagr(d,actions=None): return cagr_series(split_adjusted_price(d,actions))[0]
def total_return_cagr(d):
    s=d['Adj Close'] if 'Adj Close' in d and d['Adj Close'].notna().sum()>=2 else d.Close
    return cagr_series(s)[0]

def pivot_swings(d,left=3,right=3,min_prominence=0.04,max_age_bars=260):
    x=clean(d).iloc[:-1].copy()
    if len(x)<2*left+2*right+20:return []
    x=x.iloc[-max_age_bars:]
    highs=x.High.values; lows=x.Low.values; idx=x.index; swings=[]
    for i in range(left,len(x)-right):
        h=float(highs[i]); lo=float(lows[i])
        if not np.isfinite(h) or not np.isfinite(lo): continue
        if h>=max(highs[i-left:i]) and h>=max(highs[i+1:i+right+1]):
            neighborhood=float(np.nanmedian(np.r_[highs[i-left:i], highs[i+1:i+right+1]]))
            if neighborhood>0 and abs(h-neighborhood)/neighborhood>=min_prominence: swings.append((idx[i],h,'H'))
        if lo<=min(lows[i-left:i]) and lo<=min(lows[i+1:i+right+1]):
            neighborhood=float(np.nanmedian(np.r_[lows[i-left:i], lows[i+1:i+right+1]]))
            if neighborhood>0 and abs(lo-neighborhood)/neighborhood>=min_prominence: swings.append((idx[i],lo,'L'))
    swings.sort(key=lambda z:z[0])
    # keep the more extreme pivot when adjacent pivots have the same type
    filtered=[]
    for s in swings:
        if filtered and filtered[-1][2]==s[2]:
            if s[2]=='H' and s[1]>=filtered[-1][1]: filtered[-1]=s
            elif s[2]=='L' and s[1]<=filtered[-1][1]: filtered[-1]=s
        else: filtered.append(s)
    return filtered

def fibonacci_targets(d):
    swings=pivot_swings(d)
    if len(swings)<2:return []
    last=float(d.Close.iloc[-1]); direction=None; a=b=None
    for i in range(len(swings)-1,0,-1):
        x,y=swings[i-1],swings[i]
        if x[2]=='L' and y[2]=='H' and y[1]>x[1]: a,b,direction=x,y,'up'; break
        if x[2]=='H' and y[2]=='L' and x[1]>y[1]: a,b,direction=x,y,'down'; break
    if direction!='up':
        return []  # strategy is long-only; down swings are reported structurally, never as long targets
    lo,hi=a[1],b[1]; rng=hi-lo
    if rng/lo < .04:return []
    vals=[(hi+rng*.272,'Fib 127.2%'),(hi+rng*.618,'Fib 161.8%')]
    return [(v,n) for v,n in vals if v>last]

def fibs(d): return fibonacci_targets(d)

def next_resistance(d,price):
    x=d.iloc[:-1]; levels=[]
    if 'RES20' in d and pd.notna(d.RES20.iloc[-1]): levels.append(float(d.RES20.iloc[-1]))
    swings=pivot_swings(d)
    levels += [v for _,v,t in swings if t=='H' and v>price]
    return min(levels) if levels else None

def technical(d,w,m):
    if len(d)<220 or len(w)<60 or len(m)<24:return None
    z=d.iloc[-1];score=0;score+=10 if z.Close>z.EMA50 else 0;score+=8 if 45<=z.RSI<=68 else 0;score+=8 if z.MACD>z.MACD_SIGNAL else 0;score+=8 if z.EMA20>z.EMA50 else 0;score+=8 if z.Volume>=1.2*z.VOL20 else 0;score+=4 if z.ADX>=20 else 0;score+=4 if z.OBV>d.OBV.iloc[-21] else 0
    score+=10 if w.Close.iloc[-1]>ema(w.Close,20).iloc[-1] else 0;score+=5 if ema(w.Close,20).iloc[-1]>ema(w.Close,50).iloc[-1] else 0;score+=5 if m.Close.iloc[-1]>ema(m.Close,12).iloc[-1] else 0;score+=5 if ema(m.Close,12).iloc[-1]>ema(m.Close,24).iloc[-1] else 0
    breakout=pd.notna(z.RES20) and z.Close>z.RES20*1.003 and z.Volume>=1.3*z.VOL20;pullback=pd.notna(z.EMA20) and pd.notna(z.SUP20) and z.Close>=z.SUP20 and z.Close<=z.EMA20*1.04;score+=10 if breakout else (7 if pullback else 0);score+=5 if z.RET20>0 else 0
    stop=max(.01,min(float(z.SUP20)-.01*float(z.Close) if pd.notna(z.SUP20) else float(z.Close)-2*float(z.ATR),float(z.Close)-2*float(z.ATR))); risk=max(0,float(z.Close)-stop)
    nr=next_resistance(d,float(z.Close)); rr=(nr-float(z.Close))/risk if risk>0 and nr and nr>z.Close else 0
    score+=10 if rr>=2 else (7 if rr>=1.5 else (4 if rr>=1 else 0))
    return min(100,int(score)),breakout,float(rr),nr

def setup(d):
    z=d.iloc[-1];p=float(z.Close);a=float(z.ATR) if pd.notna(z.ATR) and float(z.ATR)>0 else p*.03
    sup=float(z.SUP20) if pd.notna(z.SUP20) else p-2*a
    low=max(sup,p-.5*a); high=max(low,p+.15*a); entry=(low+high)/2
    min_distance=max(a, .02*entry)
    support_stop=sup-.25*a
    stop=min(entry-min_distance, support_stop)
    if stop<=0 or stop>=entry: stop=entry-min_distance
    risk=max(entry-stop, min_distance)
    nr=next_resistance(d,entry); candidates=[]
    if nr and nr>entry+risk: candidates.append((nr,'Next resistance / pivot'))
    candidates += fibonacci_targets(d)
    targets=[]
    for v,n in sorted(candidates):
        if v>entry+risk and all(abs(v-t[0])>.01*entry for t in targets):
            targets.append((v,n))
        if len(targets)==3:break
    for mult in (1.5,2.5,3.5):
        if len(targets)==3: break
        v=entry+mult*risk
        if all(abs(v-t[0])>.01*entry for t in targets): targets.append((v,f'Risk/ATR {mult:.1f}x'))
    return low,high,stop,targets[:3]

def position(capital,risk_pct,entry,stop):
    if capital<=0 or risk_pct<=0 or entry<=stop:return 0,0,0
    risk=capital*risk_pct/100;shares=math.floor(min(risk/(entry-stop), capital/entry));return shares,shares*entry,shares*(entry-stop)

def sector_class(s):
    s=(s or '').lower();
    if any(k in s for k in ['bank','insurance','financial services','capital markets']):return 'financial'
    if 'real estate' in s or 'property' in s:return 'real_estate'
    return 'general'

def normalize_dividend_yield(value):
    if value is None:return None
    try:
        v=float(value)
        if not np.isfinite(v) or v<0:return None
        if v>1.0 and v<=100.0:v/=100.0
        return v if v<=1.0 else None
    except Exception:return None

def freshness_score(last_date, now=None, good_days=3, stale_days=30):
    if last_date is None:return 0.0
    try:
        d=pd.Timestamp(last_date); now=pd.Timestamp(now or datetime.now(timezone.utc));
        if d.tzinfo is None and now.tzinfo is not None:d=d.tz_localize(now.tz)
        age=max(0,(now-d).total_seconds()/86400)
        if age<=good_days:return 100.0
        if age>=stale_days:return 0.0
        return 100*(stale_days-age)/(stale_days-good_days)
    except Exception:return 0.0

def _valid_metric(v):
    try:
        return v is not None and np.isfinite(float(v))
    except Exception:
        return False

def _metric_quality(fields):
    valid=sum(_valid_metric(v) for v in fields.values())
    total=len(fields)
    if total == 0:
        return 0.0, 0
    return round(100.0*valid/total,1), valid

def fundamentals(client,ticker):
    retrieved=datetime.now(timezone.utc)
    info,info_status,info_error=client.info(ticker); inc,income_status,income_error=client.income_stmt(ticker)
    info_ok=info_status=='ok'; income_ok=income_status=='ok'
    def series(names):
        for n in names:
            if n in inc.index:
                return pd.to_numeric(inc.loc[n],errors='coerce').dropna().sort_index()
        return pd.Series(dtype=float)
    rev,rs=cagr_series(series(['Total Revenue','Operating Revenue']))
    eps,es=cagr_series(series(['Diluted EPS','Basic EPS']))
    get=lambda k: info.get(k)
    dy=normalize_dividend_yield(get('dividendYield'))
    fin_date=None
    if income_ok:
        try:
            dates=[pd.Timestamp(c) for c in inc.columns if pd.Timestamp(c)<=pd.Timestamp(retrieved)]
            fin_date=max(dates) if dates else None
        except Exception:
            fin_date=None
    fields={
        'roe':get('returnOnEquity'),'margin':get('profitMargins'),
        'pe':get('trailingPE') if _valid_metric(get('trailingPE')) and float(get('trailingPE'))>0 else None,
        'pb':get('priceToBook') if _valid_metric(get('priceToBook')) and float(get('priceToBook'))>0 else None,
        'de':get('debtToEquity') if _valid_metric(get('debtToEquity')) and float(get('debtToEquity'))>=0 else None,
        'dy':dy,'revenue_cagr':rev,'eps_cagr':eps,
        'ev_ebitda':get('enterpriseToEbitda') if _valid_metric(get('enterpriseToEbitda')) and float(get('enterpriseToEbitda'))>0 else None,
        'ebitda':get('ebitda') if _valid_metric(get('ebitda')) and float(get('ebitda'))>0 else None,
        'total_debt':get('totalDebt') if _valid_metric(get('totalDebt')) and float(get('totalDebt'))>=0 else None,
        'cash':get('totalCash') if _valid_metric(get('totalCash')) and float(get('totalCash'))>=0 else None,
        'shares_outstanding':get('sharesOutstanding'),'trailing_eps':get('trailingEps'),
        'quote_currency':get('currency') or get('quoteCurrency'),'financial_currency':get('financialCurrency') or get('currency'),'book_value_per_share':get('bookValue') if _valid_metric(get('bookValue')) and float(get('bookValue'))>0 else None
    }
    metric_fields={k:v for k,v in fields.items() if k not in {'quote_currency','financial_currency'}}
    fundamentals_quality,usable=_metric_quality(metric_fields)
    strength_parts=[]
    if _valid_metric(fields['roe']): strength_parts.append(float(fields['roe'])/0.20)
    if _valid_metric(fields['margin']): strength_parts.append(float(fields['margin'])/0.20)
    if _valid_metric(fields['revenue_cagr']): strength_parts.append((float(fields['revenue_cagr'])+0.05)/0.20)
    if _valid_metric(fields['eps_cagr']): strength_parts.append((float(fields['eps_cagr'])+0.05)/0.25)
    if _valid_metric(fields['de']): strength_parts.append(1.0-float(fields['de'])/150.0)
    if _valid_metric(fields['dy']): strength_parts.append(float(fields['dy'])/0.08)
    fundamentals_strength=round(float(np.clip(np.mean(np.clip(strength_parts,0,1))*100 if strength_parts else 0,0,100)),1)
    # Retrieved is not enough: require at least 2 usable valuation/quality metrics.
    fundamentals_ok=(info_ok or income_ok) and usable>=2
    info_date=None
    try:
        q=get('mostRecentQuarter')
        if q is not None: info_date=pd.Timestamp(q)
    except Exception:
        info_date=None
    return {'sector':get('sector') or 'Unknown','industry':get('industry') or 'Unknown','sector_class':sector_class(get('sector')),'roe':fields['roe'],'margin':fields['margin'],
            'pe':fields['pe'],'pb':fields['pb'],'de':fields['de'],'dy':dy,'dy_pct':dy*100 if dy is not None else None,
            'revenue_cagr':rev,'eps_cagr':eps,'eps_status':es,'revenue_status':rs,'ev_ebitda':fields['ev_ebitda'],'ebitda':fields['ebitda'],'total_debt':fields['total_debt'],'cash':fields['cash'],'net_debt':(float(fields['total_debt'])-float(fields['cash'])) if _valid_metric(fields['total_debt']) and _valid_metric(fields['cash']) else None,'shares_outstanding':fields['shares_outstanding'],'trailing_eps':fields['trailing_eps'],'book_value_per_share':fields['book_value_per_share'],
            'quote_currency':fields['quote_currency'],'financial_currency':fields['financial_currency'],
            'currency_consistent':bool(fields['quote_currency'] and fields['financial_currency'] and str(fields['quote_currency']).upper()==str(fields['financial_currency']).upper()),
            'fundamentals_unit_scale':'absolute_yahoo' if info_ok or income_ok else 'unknown',
            'financial_last_date':fin_date,'info_most_recent_date':info_date,
            'fundamentals_retrieved_at_utc':retrieved,'info_ok':info_ok,'income_ok':income_ok,'fundamentals_ok':fundamentals_ok,
            'fundamentals_quality':fundamentals_quality,'fundamentals_strength':fundamentals_strength,'fundamentals_usable_metrics':usable,'fundamentals_metric_count':len(fields),
            'info_status':info_status,'info_error':info_error,'income_status':income_status,'income_error':income_error,'actions_status':'not_requested','actions_error':'','fundamentals_completeness':fundamentals_quality}

# EGX calendar: Sunday-Thursday baseline plus maintained full-close dates.
# Additional dates can be supplied via EGX_HOLIDAYS_FILE as JSON/CSV for future official schedules.
EGX_DEFAULT_HOLIDAYS = {
    2026: {"2026-01-01","2026-01-07","2026-01-25","2026-03-19","2026-03-22","2026-03-23",
           "2026-04-12","2026-04-13","2026-05-27","2026-05-28","2026-06-17","2026-06-30",
           "2026-07-23","2026-08-26","2026-10-06"},
}
def _load_egx_holidays():
    dates=set()
    for vals in EGX_DEFAULT_HOLIDAYS.values(): dates.update(vals)
    path=os.getenv('EGX_HOLIDAYS_FILE','').strip()
    if path and os.path.exists(path):
        try:
            if path.lower().endswith('.json'):
                obj=json.load(open(path,encoding='utf-8'))
                if isinstance(obj,dict):
                    for vals in obj.values():
                        if isinstance(vals,list): dates.update(str(x)[:10] for x in vals)
                elif isinstance(obj,list): dates.update(str(x)[:10] for x in obj)
            else:
                raw=pd.read_csv(path)
                col='date' if 'date' in raw.columns else raw.columns[0]
                dates.update(pd.to_datetime(raw[col],errors='coerce').dropna().dt.strftime('%Y-%m-%d'))
        except Exception as exc:
            logger.warning("Could not load EGX holiday file: %s", exc)
    return dates
EGX_HOLIDAYS=_load_egx_holidays()
EGX_CALENDAR_SOURCE='maintained_2026_snapshot + EGX_HOLIDAYS_FILE override'
EGX_CALENDAR_VERSION='2026.1'

def egx_expected_sessions(start,end):
    start=pd.Timestamp(start).normalize(); end=pd.Timestamp(end).normalize()
    days=pd.date_range(start,end,freq='D')
    return [d for d in days if d.weekday() in (6,0,1,2,3) and d.strftime('%Y-%m-%d') not in EGX_HOLIDAYS]

def trading_gap_stats(d):
    """EGX calendar-aware expected sessions; Friday/Saturday excluded and known full closures removed."""
    if d is None or d.empty: return {'expected_sessions':0,'observed_sessions':0,'missing_sessions':0,'gap_ratio':1.0,'calendar':EGX_CALENDAR_SOURCE}
    try:
        start=pd.Timestamp(d.index.min()).normalize(); end=pd.Timestamp(d.index.max()).normalize()
        expected=egx_expected_sessions(start,end)
        observed_idx=pd.DatetimeIndex(d.index).tz_localize(None).normalize().unique()
        expected_set={x.date() for x in expected}; observed_set={x.date() for x in observed_idx}
        # Count only bars that fall on expected EGX sessions; weekend/closure rows
        # from generic mock or external providers are not EGX observations.
        observed_set=observed_set & expected_set
        observed=len(observed_set)
        missing=len(expected_set-observed_set); ratio=missing/max(1,len(expected_set))
        return {'expected_sessions':int(len(expected_set)),'observed_sessions':int(observed),'missing_sessions':int(missing),'gap_ratio':float(ratio),'calendar':EGX_CALENDAR_SOURCE}
    except Exception:
        return {'expected_sessions':0,'observed_sessions':len(d),'missing_sessions':0,'gap_ratio':0.0,'calendar':'fallback'}


def history_sufficiency(d):
    if d is None or d.empty: return {'daily':0.0,'weekly':0.0,'monthly':0.0,'overall':0.0}
    valid=d.dropna(subset=[c for c in ['Open','High','Low','Close','Volume'] if c in d])
    w,m=completed_tf(valid)
    vals=[min(1,len(valid)/220),min(1,len(w)/60),min(1,len(m)/24)]
    return {'daily':round(vals[0]*100,1),'weekly':round(vals[1]*100,1),'monthly':round(vals[2]*100,1),'overall':round(float(np.mean(vals)*100),1)}

def history_quality(d,repair_status='success'):
    """Integrity/coverage quality only. Length is NOT quality; it is sufficiency."""
    if d is None or d.empty:return 0.0
    cols=['Open','High','Low','Close','Volume']
    completeness=float(1-d[cols].isna().mean().mean()) if all(c in d for c in cols) else 0.0
    valid=d.dropna(subset=cols)
    gaps=trading_gap_stats(valid)
    gap_factor=max(0.0,1.0-0.90*gaps['gap_ratio'])
    repair_factor=1.0 if repair_status=='success' else .95 if str(repair_status).startswith('unsupported_repair_fallback') else .75 if str(repair_status).startswith('repair_failed_fallback') else .50
    # Recent bars must be valid; reject a superficially non-empty history ending in NaN.
    recent_ok=1.0 if len(valid)>=1 and pd.notna(valid['Close'].iloc[-1]) and pd.notna(valid['High'].iloc[-1]) and pd.notna(valid['Low'].iloc[-1]) else 0.0
    q=100*(.60*completeness+.25*gap_factor+.15*recent_ok)*repair_factor
    return round(float(np.clip(q,0,100)),1)

def quality(d,f,fair_available=False,peer_count=0,price_last_date=None,history_quality=None):
    hq=float(history_quality if history_quality is not None else history_quality_fn(d))
    price_f=freshness_score(price_last_date)
    fund_f=freshness_score(f.get('financial_last_date'),good_days=45,stale_days=365)
    fq=float(f.get('fundamentals_completeness',f.get('fundamentals_quality',0)))
    strength=float(f.get('fundamentals_strength',50))
    peer_cov=min(100,peer_count/10*100)
    # Data quality is about data reliability/completeness/freshness, NOT whether valuation exists.
    q=.35*hq+.20*price_f+.15*fund_f+.15*fq+.10*strength+.05*peer_cov
    return round(float(np.clip(q,0,100)),1)

def history_quality_fn(d): return history_quality(d,'success')

def peer_eligibility(row):
    reasons=[]
    price=row.get('price')
    sector=row.get('sector_class')
    if not _valid_metric(price) or float(price)<=0: reasons.append('invalid_price')
    if not sector or str(sector)=='Unknown': reasons.append('missing_sector_class')
    metric_ok=False
    if _valid_metric(row.get('pe')) and float(row.get('pe'))>0: metric_ok=True
    if _valid_metric(row.get('pb')) and float(row.get('pb'))>0: metric_ok=True
    ev_ready=(_valid_metric(row.get('ev_ebitda')) and float(row.get('ev_ebitda'))>0 and
              _valid_metric(row.get('ebitda')) and float(row.get('ebitda'))>0 and
              _valid_metric(row.get('shares_outstanding')) and float(row.get('shares_outstanding'))>0 and
              bool(row.get('currency_consistent',False)))
    if ev_ready: metric_ok=True
    if not metric_ok: reasons.append('no_usable_valuation_multiple')
    eligible=not reasons
    return eligible, ('eligible' if eligible else ';'.join(reasons))

def fair_value(row,peer_universe):
    peers=peer_universe[(peer_universe.sector_class==row.sector_class)&(peer_universe.symbol!=row.symbol)].copy()
    # Prefer same Industry when available; fall back to sector class only if industry coverage is insufficient.
    industry=str(row.get('industry','Unknown'))
    if industry and industry!='Unknown' and 'industry' in peers.columns:
        same=peers[peers.industry.astype(str).eq(industry)]
        if len(same)>=3: peers=same
    profiles={
        'financial':[('pb',.55),('pe',.30),('ev_ebitda',.15)],
        'real_estate':[('pb',.55),('pe',.25),('ev_ebitda',.20)],
        'general':[('pe',.45),('ev_ebitda',.35),('pb',.20)]
    }
    estimates=[];details=[];peer_counts=[]
    for metric,weight in profiles.get(row.sector_class,profiles['general']):
        current=row.get(metric)
        if metric not in peers or not _valid_metric(current) or float(current)<=0: continue
        pvals=pd.to_numeric(peers[metric],errors='coerce')
        pvals=pvals[np.isfinite(pvals)&(pvals>0)]
        if len(pvals)<3: continue
        q1,q3=pvals.quantile([.25,.75]);iqr=q3-q1
        if np.isfinite(iqr) and iqr>0:
            pvals=pvals[(pvals>=q1-1.5*iqr)&(pvals<=q3+1.5*iqr)]
        if len(pvals)<3: continue
        bench=float(pvals.median()); implied=None
        if metric=='ev_ebitda':
            # EV/EBITDA must be converted to equity value before dividing by shares.
            ebitda=row.get('ebitda'); net_debt=row.get('net_debt'); shares=row.get('shares_outstanding')
            if not (_valid_metric(ebitda) and _valid_metric(net_debt) and _valid_metric(shares)) or float(ebitda)<=0 or float(shares)<=0 or not bool(row.get('currency_consistent',False)):
                continue
            implied=(float(ebitda)*bench-float(net_debt))/float(shares)
        elif metric=='pe':
            eps=row.get('trailing_eps')
            if not _valid_metric(eps) or float(eps)<=0:
                eps=float(row.price)/float(current)
            implied=float(eps)*bench
        elif metric=='pb':
            book_per_share=row.get('book_value_per_share')
            if not _valid_metric(book_per_share) or float(book_per_share)<=0:
                book_per_share=float(row.price)/float(current)
            implied=float(book_per_share)*bench
        if implied is not None and np.isfinite(implied) and implied>0:
            estimates.append((implied,weight));peer_counts.append(len(pvals))
            details.append(f'{metric.upper()} median={bench:.2f} ({len(pvals)} peers)')
    if not estimates:return None,'N/A (<3 usable independent peers)',0,None,0,'Low'
    total=sum(w for _,w in estimates); implied=sum(v*w for v,w in estimates)/total
    mos=(implied-float(row.price))/implied if implied>0 else None
    metric_count=len(estimates); conf='High' if metric_count>=3 else ('Medium' if metric_count==2 else 'Low')
    return round(implied,2),'Sector-aware robust weighted multiples: '+'; '.join(details),int(max(peer_counts)),mos,metric_count,conf

def inv_score(r):
    if r.data_quality<50:return None
    growth=[]
    if pd.notna(r.revenue_cagr):growth.append(np.clip((r.revenue_cagr+.05)/.20,0,1))
    if pd.notna(r.eps_cagr):growth.append(np.clip((r.eps_cagr+.05)/.25,0,1))
    s=25*np.mean(growth) if growth else 0;quality_vals=[]
    if pd.notna(r.roe):quality_vals.append(np.clip(r.roe/.20,0,1))
    if pd.notna(r.margin):quality_vals.append(np.clip(r.margin/.20,0,1))
    s+=25*np.mean(quality_vals) if quality_vals else 0;s+=25*np.clip((r.mos+.20)/.50,0,1) if pd.notna(r.mos) else 0;balance=[]
    if r.sector_class!='financial' and pd.notna(r.de):balance.append(np.clip(1-r.de/150,0,1))
    if pd.notna(r.dy):balance.append(np.clip(r.dy/.08,0,1))
    s+=15*np.mean(balance) if balance else 0;s+=10*r.technical_score/100;return round(float(np.clip(s,0,100)),2)

def analyze_one(client,sym,capital,risk):
    ticker=client.ticker(sym); d,repair_status,history_error=client.history(ticker); f=fundamentals(client,ticker); price_last=d.index[-1] if not d.empty else None
    hq=history_quality(d,repair_status) if not d.empty else 0.0
    gaps=trading_gap_stats(d)
    # Split data is already returned by history(actions=True). Do not issue a second actions request.
    split_actions=d[['Stock Splits']].copy() if (not d.empty and 'Stock Splits' in d.columns) else pd.DataFrame()
    actions_status='embedded_in_history_with_splits' if not split_actions.empty and split_actions['Stock Splits'].fillna(0).ne(0).any() else ('embedded_in_history_empty' if not split_actions.empty else 'fallback_required')
    actions_error=''
    if split_actions.empty:
        # Only fallback to a separate actions request when history did not expose split data.
        split_actions, actions_status, actions_error=client.actions(ticker)
        if not split_actions.empty and 'Stock Splits' in split_actions.columns:
            actions_status='fallback_used'
        else:
            actions_status='fallback_empty'
    base={'symbol':sym,**f,'history_ok':not d.empty,'history_quality':hq,'history_sufficiency':history_sufficiency(d),'price_last_date':price_last,'repair_status':repair_status,
          'history_error':history_error,'actions_status':actions_status,'actions_error':actions_error,'technical_data_sufficient':False,'expected_sessions':gaps['expected_sessions'],'observed_sessions':gaps['observed_sessions'],'missing_sessions':gaps['missing_sessions'],'gap_ratio':gaps['gap_ratio']}
    if d.empty:return base
    di=add_ind(d);w,m=completed_tf(d);tech=technical(di,w,m)
    base.update({'price':float(di.Close.iloc[-1]),'total_return_cagr':total_return_cagr(d),'price_cagr':price_cagr(d,split_actions)})
    if tech is None:
        base['technical_data_sufficiency_reason']='daily>=220, completed_weekly>=60, completed_monthly>=24'
        return base
    low,high,stop,tps=setup(di);entry=(low+high)/2;sh,pv,rv=position(capital,risk,entry,stop)
    base.update({'technical_data_sufficient':True,'technical_score':tech[0],'breakout':tech[1],'rr':tech[2],'next_resistance':tech[3],'entry_low':low,'entry_high':high,'stop':stop,
                 'tp1':tps[0][0],'tp1_source':tps[0][1],'tp2':tps[1][0],'tp2_source':tps[1][1],'tp3':tps[2][0],'tp3_source':tps[2][1],
                 'shares':sh,'position_value':pv,'risk_value':rv})
    return base

def batch_history(symbols, timeout=20, years=5):
    """Fetch all EGX daily candles in one Yahoo multi-ticker request.
    This is the main speed path: technical analysis is local after the batch download.
    """
    tickers=[f'{sym}.CA' for sym in symbols]
    try:
        data=yf.download(
            tickers=tickers,
            period=f'{int(years)}y', interval='1d',
            auto_adjust=False, actions=True,
            group_by='ticker', threads=True, progress=False,
            timeout=timeout, repair=True
        )
        return data, '', 'success'
    except TypeError:
        try:
            data=yf.download(
                tickers=tickers, period=f'{int(years)}y', interval='1d',
                auto_adjust=False, actions=True,
                group_by='ticker', threads=True, progress=False,
                timeout=timeout
            )
            return data, 'repair_unsupported', 'success'
        except Exception as exc:
            return pd.DataFrame(), '', f'{type(exc).__name__}: {str(exc)[:220]}'
    except Exception as exc:
        return pd.DataFrame(), '', f'{type(exc).__name__}: {str(exc)[:220]}'


def extract_batch_symbol(data, sym, symbols):
    if data is None or getattr(data, 'empty', True):
        return pd.DataFrame()
    key=f'{sym}.CA'
    try:
        if isinstance(data.columns, pd.MultiIndex):
            lvl0=set(str(x) for x in data.columns.get_level_values(0))
            lvl1=set(str(x) for x in data.columns.get_level_values(1))
            if key in lvl0:
                d=data[key].copy()
            elif key in lvl1:
                d=data.xs(key, axis=1, level=1).copy()
            elif sym in lvl0:
                d=data[sym].copy()
            elif sym in lvl1:
                d=data.xs(sym, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            # Single-symbol fallback.
            d=data.copy()
        return clean(d)
    except Exception:
        return pd.DataFrame()


def fast_technical_scan(symbols, capital, risk, batch_data, repair_status='success'):
    rows=[]; errors=[]
    for sym in symbols:
        try:
            d=extract_batch_symbol(batch_data, sym, symbols)
            hq=history_quality(d, repair_status) if not d.empty else 0.0
            gaps=trading_gap_stats(d)
            split_actions=d[['Stock Splits']].copy() if (not d.empty and 'Stock Splits' in d.columns) else pd.DataFrame()
            if split_actions.empty:
                split_actions=None
            base={'symbol':sym,'history_ok':not d.empty,'history_quality':hq,
                  'history_sufficiency':history_sufficiency(d),'price_last_date':d.index[-1] if not d.empty else None,
                  'repair_status':repair_status,'history_error':'' if not d.empty else 'No batch history returned',
                  'actions_status':'embedded_in_history' if split_actions is not None else 'not_available',
                  'actions_error':'','technical_data_sufficient':False,
                  'expected_sessions':gaps['expected_sessions'],'observed_sessions':gaps['observed_sessions'],
                  'missing_sessions':gaps['missing_sessions'],'gap_ratio':gaps['gap_ratio']}
            if d.empty:
                rows.append(base); continue
            di=add_ind(d); w,m=completed_tf(d); tech=technical(di,w,m)
            base.update({'price':float(di.Close.iloc[-1]),
                         'total_return_cagr':total_return_cagr(d),
                         'price_cagr':price_cagr(d,split_actions)})
            if tech is None:
                base['technical_data_sufficiency_reason']='daily>=220, completed_weekly>=60, completed_monthly>=24'
            else:
                low,high,stop,tps=setup(di); entry=(low+high)/2
                sh,pv,rv=position(capital,risk,entry,stop)
                base.update({'technical_data_sufficient':True,'technical_score':tech[0],
                    'breakout':tech[1],'rr':tech[2],'next_resistance':tech[3],
                    'entry_low':low,'entry_high':high,'stop':stop,
                    'tp1':tps[0][0],'tp1_source':tps[0][1],'tp2':tps[1][0],'tp2_source':tps[1][1],
                    'tp3':tps[2][0],'tp3_source':tps[2][1],
                    'shares':sh,'position_value':pv,'risk_value':rv})
            rows.append(base)
        except Exception as exc:
            errors.append((sym,type(exc).__name__,str(exc)[:160]))
    return rows,errors


def scan_core(client, capital, risk, workers=1, symbols=None, fundamentals_limit=10, years=5):
    """FAST DATA mode: batch candles for the whole universe, then fundamentals only for
    the best technical candidates. This avoids 2-3 Yahoo requests per stock.
    """
    symbols=symbols or SYMBOLS
    started=datetime.now(timezone.utc); errors=[]
    batch_data, batch_status, batch_error=batch_history(symbols, timeout=getattr(client,'timeout',20), years=years)
    if batch_error:
        errors.append(('BATCH','history',batch_error))
        return pd.DataFrame(),errors,{'universe_size':len(symbols),'batch_history_ok':False,'batch_history_error':batch_error}

    raw,tech_errors=fast_technical_scan(symbols,capital,risk,batch_data,batch_status or 'success')
    errors.extend(tech_errors)
    tech_rows=[r for r in raw if 'technical_score' in r]
    tech_rank=sorted(tech_rows,key=lambda r:(r.get('technical_score',-1),r.get('rr',0),r.get('history_quality',0)),reverse=True)
    fund_symbols=[r['symbol'] for r in tech_rank[:max(1,int(fundamentals_limit))]]

    # Fundamentals are intentionally limited to the strongest technical candidates.
    mgr=client
    def fund_one(sym):
        try:
            t=mgr.ticker(sym); f=fundamentals(mgr,t); return sym,f,None
        except Exception as exc:
            return sym,{},f'{type(exc).__name__}: {str(exc)[:220]}'
    if fund_symbols:
        with ThreadPoolExecutor(max_workers=max(1,min(int(workers),len(fund_symbols)))) as ex:
            futs=[ex.submit(fund_one,s) for s in fund_symbols]
            for fut in as_completed(futs):
                sym,f,err=fut.result()
                for r in raw:
                    if r.get('symbol')==sym:
                        r.update(f)
                        r['fundamentals_requested']=True
                        r['fundamentals_ok']=bool(f.get('fundamentals_ok'))
                        if err:
                            r['info_error']=err; r['income_error']=err
                        break
    for r in raw:
        if 'fundamentals_requested' not in r:
            r.update({'sector':'Not analyzed','industry':'Not analyzed','sector_class':'Unknown',
                      'fundamentals_ok':False,'fundamentals_quality':0.0,'fundamentals_strength':0.0,
                      'fundamentals_completeness':0.0,'fundamentals_requested':False,
                      'info_status':'skipped','income_status':'skipped','info_error':'Skipped for speed: not in top technical candidates',
                      'income_error':'Skipped for speed: not in top technical candidates','financial_last_date':None,
                      'info_most_recent_date':None,'fundamentals_retrieved_at_utc':None})
        eligible,reason=peer_eligibility(r)
        r['peer_eligible']=eligible; r['peer_eligibility_reason']=reason

    successful_fund=[r for r in raw if r.get('fundamentals_ok')]
    eligible_peers=[r for r in successful_fund if r.get('peer_eligible')]
    peer_cols=['symbol','sector_class','sector','industry','pe','pb','ev_ebitda','ebitda','net_debt','shares_outstanding','trailing_eps','book_value_per_share','price','currency_consistent','quote_currency','financial_currency','peer_eligible','peer_eligibility_reason']
    peer_universe=pd.DataFrame(eligible_peers)[[c for c in peer_cols if c in pd.DataFrame(eligible_peers).columns]].drop_duplicates('symbol') if eligible_peers else pd.DataFrame(columns=peer_cols)
    df=pd.DataFrame(tech_rows)
    coverage={'universe_size':len(symbols),'batch_history_ok':True,'batch_mode':True,
        'price_coverage':sum(bool(r.get('history_ok')) for r in raw),
        'fundamentals_coverage':len(successful_fund),'fundamentals_attempted':len(fund_symbols),
        'fundamentals_skipped':max(0,len(symbols)-len(fund_symbols)),
        'technical_coverage':len(tech_rows),'fundamentals_top_n':len(fund_symbols),'history_years':int(years),
        'top_technical_symbols':fund_symbols,'peer_eligible_coverage':len(eligible_peers),
        'peer_universe_size':len(peer_universe),'fundamentals_quality_avg':round(float(np.mean([r.get('fundamentals_quality',0) for r in successful_fund])) if successful_fund else 0,1),
        'fundamentals_strength_avg':round(float(np.mean([r.get('fundamentals_strength',0) for r in successful_fund])) if successful_fund else 0,1),
        'egx_calendar':EGX_CALENDAR_SOURCE,'egx_calendar_version':EGX_CALENDAR_VERSION,
        'scan_started_utc':started.isoformat(),'backend':('curl_cffi' if CURL_CFFI_AVAILABLE else 'requests'),
        'python_version':platform.python_version(),'pandas_version':pd.__version__,'yfinance_version':getattr(yf,'__version__','unknown')}
    gate=getattr(client,'gate',None)
    if gate is not None:
        coverage.update({'transport_attempts':gate.transport_attempts,'data_requests':gate.data_requests,'cookie_requests':gate.cookie_requests,'crumb_requests':gate.crumb_requests,
            'http_retries':gate.retry_count,'http_429':gate.rate_limit_count,'http_last_cooldown_seconds':gate.last_cooldown_seconds})
    if df.empty:
        coverage['scan_finished_utc']=datetime.now(timezone.utc).isoformat(); return df,errors,coverage
    fv=[fair_value(r,peer_universe) for _,r in df.iterrows()]
    fv_df=pd.DataFrame(fv,index=df.index,columns=['fair_value','fair_method','peer_count','mos','valuation_metrics_used','valuation_confidence'])
    df=fv_df.join(df); df['peer_universe_size']=len(peer_universe)
    df['fundamentals_freshness_score']=df['financial_last_date'].apply(lambda x:freshness_score(x,good_days=45,stale_days=365))
    df['price_freshness_score']=df['price_last_date'].apply(freshness_score)
    df['data_quality']=df.apply(lambda r:quality(pd.DataFrame(),r.to_dict(),pd.notna(r.fair_value),int(r.peer_count),r.price_last_date,r.history_quality),axis=1)
    df['valuation_quality']=df.apply(lambda r:round(float(np.clip((25*min(1,int(r.valuation_metrics_used)/3)+25*min(1,int(r.peer_count)/5)+25*(1 if pd.notna(r.fair_value) else 0)+25*(1 if r.valuation_confidence=='High' else .6 if r.valuation_confidence=='Medium' else .25)),0,100)),1),axis=1)
    # Keep the full technical universe; investment score is only fully enriched for top-N.
    df['investment_score']=df.apply(inv_score,axis=1)
    df['confidence']=df.apply(lambda r:'High' if r.data_quality>=85 and pd.notna(r.fair_value) and r.peer_count>=3 and r.fundamentals_quality>=70 and r.valuation_metrics_used>=2 and r.valuation_confidence=='High' else ('Medium' if r.data_quality>=65 and r.fundamentals_quality>=50 and (pd.isna(r.fair_value) or r.valuation_metrics_used>=1) else 'Low'),axis=1)
    weights={'price_cagr':.15,'total_return_cagr':.35,'revenue_cagr':.20,'eps_cagr':.30}
    def wc(row):
        a=[(float(row[k]),w) for k,w in weights.items() if pd.notna(row.get(k))]
        return np.nan if len(a)<3 else float(np.average([v for v,_ in a],weights=[w for _,w in a]))
    df['blended_historical_cagr']=df.apply(wc,axis=1)
    coverage['scan_finished_utc']=datetime.now(timezone.utc).isoformat(); coverage['history_calendar_holidays_loaded']=len(EGX_HOLIDAYS)
    return df.sort_values(['investment_score','technical_score'],ascending=False,na_position='last'),errors,coverage

def live_yahoo_smoke_test(symbol='COMI'):
    """Real opt-in Yahoo integration gate.
    Exercises one shared real Session, history/info/income/actions, then 2-3
    symbols through the worker pool. External Yahoo state is reported honestly.
    """
    if os.getenv('EGX_LIVE_SMOKE','0')!='1':
        return {'enabled':False,'status':'skipped','reason':'set EGX_LIVE_SMOKE=1'}
    symbols=list(dict.fromkeys([symbol,'MFPC','ABUK']))
    mgr=YahooHTTPManager(interval=1.0,retries=1,timeout=20)
    results=[]
    def one(sym):
        try:
            t=mgr.ticker(sym)
            d,hs,he=mgr.history(t)
            info,is_,ie=mgr.info(t)
            inc,ins,ine=mgr.income_stmt(t)
            act,as_,ae=mgr.actions(t)
            ohlc=not d.empty and all(c in d.columns for c in ['Open','High','Low','Close'])
            return {'symbol':sym,'history_status':hs,'history_rows':len(d),'history_ok':ohlc,
                    'info_status':is_,'info_ok':bool(info),'income_status':ins,'income_ok':not inc.empty,
                    'actions_status':as_,'actions_ok':not act.empty,'errors':'; '.join(x for x in [he,ie,ine,ae] if x)}
        except Exception as exc:
            return {'symbol':sym,'history_status':'exception','history_rows':0,'history_ok':False,
                    'info_status':'exception','info_ok':False,'income_status':'exception','income_ok':False,
                    'actions_status':'exception','actions_ok':False,'errors':f'{type(exc).__name__}: {str(exc)[:220]}'}
    with ThreadPoolExecutor(max_workers=min(3,len(symbols))) as ex:
        futs=[ex.submit(one,s) for s in symbols]
        for f in as_completed(futs): results.append(f.result())
    results=sorted(results,key=lambda x:symbols.index(x['symbol']))
    primary=results[0]
    all_ok=all(r['history_ok'] and r['info_ok'] and r['income_ok'] and r['actions_ok'] for r in results)
    backend='curl_cffi' if CURL_CFFI_AVAILABLE and os.getenv('EGX_FORCE_REQUESTS','0')!='1' else 'requests'
    return {'enabled':True,'status':'ok' if all_ok else 'failed','symbol':symbol,'backend':backend,
            'shared_session':True,'worker_symbols':symbols,'parallel_workers':min(3,len(symbols)),
            'primary':primary,'symbols':results,'transport_attempts':mgr.gate.transport_attempts,
            'cookie_requests':mgr.gate.cookie_requests,'crumb_requests':mgr.gate.crumb_requests,
            'data_requests':mgr.gate.data_requests,'cookie_responses':mgr.gate.cookie_responses,
            'crumb_responses':mgr.gate.crumb_responses,'http_retries':mgr.gate.retry_count,
            'http_429':mgr.gate.rate_limit_count,'yfinance_internal_cookie_crumb_retry_possible':True,
            'yfinance_calls_serialized':True,'curl_retry_disabled':True,'yfinance_retry_disabled':True}

@st.cache_data(ttl=1800)
def scan(capital,risk,workers,fundamentals_limit,years):
    manager=YahooHTTPManager(interval=0.25,retries=1,timeout=15)
    return scan_core(manager,capital,risk,workers,SYMBOLS,fundamentals_limit,years)

VERSION = 'V24.0 FAST DATA/PRODUCTION'
st.set_page_config(page_title=f'EGX Analyzer {VERSION}',layout='wide');st.title(f'📈 EGX Smart Investment & Entry Analyzer {VERSION}')
with st.sidebar:
    capital=st.number_input('رأس المال بالجنيه',0.0,10_000_000.0,100_000.0,5000.0);risk=st.number_input('مخاطرة الصفقة %',.1,5.0,1.0,.1);workers=st.slider('اتصالات متوازية',1,6,4);fundamentals_limit=st.slider('عدد الأسهم للفحص المالي العميق',5,30,10);years=st.selectbox('عدد سنين الفحص التاريخي',[2,3,5,10],index=2,help='الفحص الفني الشهري الكامل يحتاج سنتين على الأقل للحصول على 24 شهرًا تقريبًا.');go=st.button('🔄 فحص السوق',type='primary')
if go:
    with st.spinner('جاري الفحص...'):df,errors,cov=scan(capital,risk,workers,fundamentals_limit,years)
    st.subheader('📡 تغطية البيانات');st.caption(f'📅 فترة الفحص التاريخي: {years} سنوات');st.json(cov)
    smoke=live_yahoo_smoke_test();
    if smoke.get('enabled'): st.write('Live Yahoo Smoke Test',smoke)
    if df.empty:st.error('لم يتم الحصول على بيانات فنية كافية من Yahoo Finance.')
    else:
        st.success(f"تم تحليل {len(df)} سهم فنيًا. Universe={cov['universe_size']} | Price={cov['price_coverage']} | Fundamentals={cov['fundamentals_coverage']} | Technical={cov['technical_coverage']} | Peer Eligible={cov.get('peer_eligible_coverage',0)} | Peers={cov['peer_universe_size']}")
        if errors:st.warning(f'تعذر تحميل {len(errors)} رمز؛ التفاصيل معروضة بدل إخفائها.')
        show=['symbol','price','sector','technical_score','investment_score','confidence','data_quality','price_cagr','total_return_cagr','revenue_cagr','eps_cagr','fair_value','fair_method','peer_count','peer_universe_size','peer_eligible','peer_eligibility_reason','mos','entry_low','entry_high','stop','rr','next_resistance','tp1','tp1_source','tp2','tp2_source','tp3','tp3_source','shares','position_value','risk_value','price_last_date','financial_last_date','info_most_recent_date','fundamentals_retrieved_at_utc','fundamentals_quality','fundamentals_strength','fundamentals_completeness','history_sufficiency','technical_data_sufficient','fundamentals_ok','fundamentals_freshness_score','valuation_quality','valuation_metrics_used','valuation_confidence','expected_sessions','observed_sessions','missing_sessions','gap_ratio','info_status','info_error','income_status','income_error','actions_status','history_error','repair_status','dy_pct']
        st.dataframe(df[[c for c in show if c in df.columns]],use_container_width=True);st.download_button('📥 CSV',df.to_csv(index=False).encode('utf-8-sig'),'egx_smart_full_report_v22.csv','text/csv')
        if errors:
            with st.expander('أخطاء التحميل'):st.dataframe(pd.DataFrame(errors,columns=['symbol','error','message']),use_container_width=True)
else:st.info('اضغط فحص السوق لبدء التحليل.')
