# app.py
# ------------------------------------------------------------
# 요구 패키지:
# pip install -U google-genai streamlit beautifulsoup4 requests pandas matplotlib numpy
# ------------------------------------------------------------

import os
import re
import json
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

# ===============================
# Matplotlib + 한글 폰트 설정
# ===============================

try:
    import matplotlib
    matplotlib.use("Agg")  # GUI 백엔드 의존 제거
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception as e:
    HAS_MPL = False
    MPL_ERR = str(e)

def set_korean_font():
    if not HAS_MPL:
        return
    candidates = [
        (r"C:\Windows\Fonts\malgun.ttf", "Malgun Gothic"),
        ("/System/Library/Fonts/AppleGothic.ttf", "AppleGothic"),
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK KR"),
        ("./fonts/NanumGothic.ttf", "NanumGothic"),
    ]
    selected = None
    for path, name in candidates:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass
            selected = name
            break
    # 폴백: 선택 실패 시에도 sans-serif에 한글 가능한 폰트 리스트 지정
    if selected:
        matplotlib.rcParams["font.family"] = selected
    matplotlib.rcParams["font.sans-serif"] = [
        selected or "Malgun Gothic",
        "AppleGothic",
        "Noto Sans CJK KR",
        "NanumGothic",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

set_korean_font()

import numpy as np

# ===============================
# Gemini SDK
# ===============================
from google import genai
from google.genai import types

# ===============================
# 0) API 키 로딩 (secrets 우선)
# ===============================
def load_api_key():
    key = None
    if hasattr(st, "secrets"):
        key = st.secrets.get("GEMINI_API_KEY", None)
    if not key:
        key = os.environ.get("GEMINI_API_KEY")
    return key

API_KEY = load_api_key()
if not API_KEY:
    st.error("❌ GEMINI_API_KEY가 없습니다. CMD에서 `setx GEMINI_API_KEY \"키값\"` 후 VS Code 재시작 "
             "또는 .streamlit/secrets.toml에 저장하세요.")
    st.stop()

# ===============================
# 1) Gemini 클라이언트
# ===============================
@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    return genai.Client(api_key=api_key)

client = get_client(API_KEY)

def call_gemini_text(prompt: str, model: str, thinking_off: bool=True) -> str:
    try:
        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0) if thinking_off else None
        )
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        text = getattr(resp, "text", "") or (
            resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else ""
        )
        return (text or "").strip()
    except Exception as e:
        return f"Gemini Error: {e}"

def call_gemini_json(prompt: str, schema: types.Schema, model: str, thinking_off: bool=True):
    try:
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_budget=0) if thinking_off else None,
        )
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        raw = getattr(resp, "text", None)
        if not raw:
            try:
                raw = resp.candidates[0].content.parts[0].text
            except Exception:
                raw = json.dumps(resp.to_dict(), ensure_ascii=False)
        return json.loads(raw), None
    except Exception as e:
        return None, f"REQUEST ERROR: {e}"

# ===============================
# 2) PDP 읽기 + 가격 힌트
# ===============================
def fetch_html(url: str):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
        r.raise_for_status()
        return r.text, None
    except Exception as e:
        return None, f"크롤링 오류: {e}"

def build_read_pack(html: str, max_body=14000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "meta", "iframe", "svg"]):
        t.decompose()
    title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()
    heads = [h.get_text(" ", strip=True) for h in soup.find_all(["h1","h2","h3","h4"]) if h.get_text(strip=True)]
    emph  = [e.get_text(" ", strip=True) for e in soup.find_all(["strong","b","em","mark"]) if e.get_text(strip=True)]
    lis   = [li.get_text(" ", strip=True) for li in soup.find_all("li") if li.get_text(strip=True)]
    body  = soup.get_text(" ", strip=True)[:max_body]
    pack = []
    if title: pack.append(f"[TITLE]\n{title}")
    if heads: pack.append("[HEADLINES]\n- " + "\n- ".join(dict.fromkeys(heads)))
    if emph:  pack.append("[EMPHASIS]\n- " + "\n- ".join(dict.fromkeys(emph)))
    if lis:   pack.append("[LIST]\n- " + "\n- ".join(lis[:300]))
    pack.append("[BODY]\n" + body)
    return "\n\n".join(pack)

PRICE_RE = re.compile(
    r'(₩[\s]*[\d,]+|\$[\s]*[\d,]+|€[\s]*[\d,]+|£[\s]*[\d,]+|[\d,]+[\s]*원)'
    r'|((일시불|판매가|정가|출고가|월[\s]*[0-9,]+[원$€£]?|무이자|할부)[^.\n\r]{0,40})',
    flags=re.IGNORECASE
)

def extract_price_hints(html: str, limit_items: int = 20):
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    matches = PRICE_RE.findall(text)
    flat = []
    for g1, g2, *_ in matches:
        cand = g1 or g2
        cand = cand.strip()
        if cand and cand not in flat:
            flat.append(cand)
        if len(flat) >= limit_items:
            break
    return flat

# ===============================
# 3) 비교 프롬프트/스키마
# ===============================
def get_compare_schema():
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "summary": types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "common": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                    "strength": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                    "weakness": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                },
                required=["common","strength","weakness"]
            ),
            "table": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "label": types.Schema(type=types.Type.STRING),
                        "ours": types.Schema(type=types.Type.STRING),
                        "theirs": types.Schema(type=types.Type.STRING),
                        "winner": types.Schema(type=types.Type.STRING),
                    },
                    required=["label","ours","theirs","winner"]
                )
            ),
            "meta": types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "brand_ours": types.Schema(type=types.Type.STRING),
                    "product_name_ours": types.Schema(type=types.Type.STRING),
                    "product_code_ours": types.Schema(type=types.Type.STRING),
                    "brand_theirs": types.Schema(type=types.Type.STRING),
                    "product_name_theirs": types.Schema(type=types.Type.STRING),
                    "product_code_theirs": types.Schema(type=types.Type.STRING),
                }
            ),
        },
        required=["summary","table"]
    )

COMPARE_JSON_INSTRUCTIONS = """
역할: 당신은 전자제품 PDP 두 페이지를 읽고, 자사 제품과 경쟁 제품을 **같은 수준의 라벨**로 비교하는 애널리스트다.

반환 형식: 반드시 JSON만 반환.
필드:
- summary.common: 두 제품의 공통점.
- summary.strength: 자사 강점.
- summary.weakness: 자사 약점.
- table: '디자인','휴대성/이동성','배터리','디스플레이','해상도/화질','밝기/색영역','오디오/스피커','OS/플랫폼','스마트 기능/연동','연결성/포트','서비스/보증','가격' 등 공통 비교 가능 항목으로 구성.
- winner: "ours" | "theirs" | "tie".

주의: 허구 금지. 모호하면 “(추정)” 표시.
"""

def make_compare_prompt(url_ours: str, pack_ours: str, price_hints_ours: list,
                        url_theirs: str, pack_theirs: str, price_hints_theirs: list) -> str:
    ours_hints_text = "- " + "\n- ".join(price_hints_ours) if price_hints_ours else "(없음)"
    theirs_hints_text = "- " + "\n- ".join(price_hints_theirs) if price_hints_theirs else "(없음)"
    return (
        COMPARE_JSON_INSTRUCTIONS
        + "\n\n[자사 제품 URL]\n" + url_ours
        + "\n\n[자사 제품 페이지 패킷]\n" + pack_ours
        + "\n\n[자사 제품 가격 힌트]\n" + ours_hints_text
        + "\n\n[경쟁 제품 URL]\n" + url_theirs
        + "\n\n[경쟁 제품 페이지 패킷]\n" + pack_theirs
        + "\n\n[경쟁 제품 가격 힌트]\n" + theirs_hints_text
    )

# ===============================
# 4) 시각화/표시 유틸
# ===============================
PILL_CSS = """
<style>
.pill-wrap{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 16px 0}
.pill{padding:10px 12px;border-radius:12px;font-size:14px;line-height:1.3}
.pill.common{background:#eef2ff;color:#1e3a8a;border:1px solid #c7d2fe}
.pill.strength{background:#e7f7e7;color:#065f46;border:1px solid #bbf7d0}
.pill.weakness{background:#fdecec;color:#7f1d1d;border:1px solid #fecaca}
.pill .tag{font-weight:700;margin-right:6px}
.small-gray{color:#6b7280 !important; font-size:13px !important;}
</style>
"""

def render_pills(title: str, items: list, kind: str):
    st.markdown(PILL_CSS, unsafe_allow_html=True)
    st.markdown(f"#### {title}")
    if not items:
        st.markdown("<div class='pill-wrap'><div class='pill common'>내용 없음</div></div>", unsafe_allow_html=True)
        return
    html = ["<div class='pill-wrap'>"]
    for it in items:
        tag = "공통" if kind=="common" else ("강점" if kind=="strength" else "약점")
        html.append(f"<div class='pill {kind}'><span class='tag'>{tag}</span>{it}</div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

def style_winner(df: pd.DataFrame):
    def highlight(row):
        base = ['']*len(row)
        try:
            w = row['winner']
            if w == 'ours':
                base[df.columns.get_loc('자사 제품')] = 'background-color:#e7f7e7'
            elif w == 'theirs':
                base[df.columns.get_loc('경쟁 제품')] = 'background-color:#e7f7e7'
        except Exception:
            pass
        return base
    return df.style.apply(highlight, axis=1)

# ===============================
# 5) 소셜 카피 스키마/프롬프트
# ===============================
def get_adcopy_schema():
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "copies": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "channel": types.Schema(type=types.Type.STRING),
                        "copy": types.Schema(type=types.Type.STRING),
                        "explanation": types.Schema(type=types.Type.STRING),
                    },
                    required=["copy","explanation"]
                )
            )
        },
        required=["copies"]
    )

AD_COPY_INSTRUCTIONS = """
역할: 디지털 카피라이터. 아래 비교 JSON을 근거로 자사 USP를 강조하는 소셜 광고 카피 3개를 한국어로 작성.
조건: 각 180자 이내, 해시태그 0~2개 허용. 과장 금지. JSON만 반환.
필드: channel, copy, explanation(의도/전략 2~3문장).
"""

def make_adcopy_prompt(compare_json: dict, url_ours: str, url_theirs: str) -> str:
    return (
        AD_COPY_INSTRUCTIONS
        + "\n\n[자사/경쟁 비교 JSON]\n"
        + json.dumps(compare_json, ensure_ascii=False)
        + "\n\n[자사 PDP]\n" + url_ours
        + "\n[경쟁 PDP]\n" + url_theirs
    )

# ===============================
# 6) 레이더(방사형) 포지셔닝 스키마/프롬프트
# ===============================
def get_radar_schema():
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "axes": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),  # 4~6개
            "ours_scores": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER)),
            "theirs_scores": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER)),
            "ours_label": types.Schema(type=types.Type.STRING),
            "theirs_label": types.Schema(type=types.Type.STRING),
            "rationales": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
            "insights": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
        },
        required=["axes","ours_scores","theirs_scores"]
    )

RADAR_INSTRUCTIONS = """
역할: 제품 포지셔닝 애널리스트. 아래 비교 JSON을 검토하고 '차별화 설명력이 높은' 4~6개 축을 선정.
요구:
- 각 축은 서로 독립적. 예: '스마트/콘텐츠 생태계', '이동성/배터리', '화질/해상도', '오디오', '가격 경쟁력' 등.
- 자사/경쟁 각각 0~10 점수로 정규화하여 배열로 반환(축 순서와 동일).
- 각 축에 대한 한 줄 근거를 'rationales' 배열로 제공(축 순서와 동일).
- JSON만 반환.
필드: axes[], ours_scores[], theirs_scores[], ours_label, theirs_label, rationales[], insights[].
주의: 허구 금지. 근거 불충분 시 “(추정)”을 명시.
"""

def make_radar_prompt(compare_json: dict, meta: dict) -> str:
    return (
        RADAR_INSTRUCTIONS
        + "\n\n[자사/경쟁 비교 JSON]\n"
        + json.dumps(compare_json, ensure_ascii=False)
        + "\n\n[메타]\n"
        + json.dumps(meta or {}, ensure_ascii=False)
    )

def plot_radar(ax, categories, values, label=None):
    N = len(categories)
    if N < 3:
        raise ValueError("레이더 차트 축은 최소 3개 이상 필요")
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    values = list(values)
    angles += angles[:1]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, marker="o", label=label)
    ax.fill(angles, values, alpha=0.15)

def radar_init(fig_size=(6.5,6.5), rmax=10):
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))
    ax.set_ylim(0, rmax)
    return fig, ax

# ===============================
# 7) UI
# ===============================
st.set_page_config(page_title="Product USP Analyzer", page_icon="⚡")
st.title("⚡ Product USP Analyzer")

# (요청) 사이드바 제거 + 고정값 사용
model = "gemini-2.5-flash"
thinking_off = True

# (요청) 텍스트 입력 기본값 설정
url_ours = st.text_input(
    "자사 제품의 **공식 PDP URL**을 입력해주세요",
    value="https://www.apple.com/kr/airpods-pro/",
)
url_theirs = st.text_input(
    "경쟁 제품의 **공식 PDP URL**을 입력해주세요",
    value="https://www.samsung.com/sec/buds/galaxy-buds3/buy/?modelCode=SM-R630NZAAKOO",
)

if st.button("두 제품 비교하기"):
    if not url_ours or not url_theirs:
        st.warning("자사/경쟁 제품의 URL을 모두 입력하세요.")
        st.stop()

    # ── 자사/경쟁 페이지 수집 ──
    with st.spinner("자사 PDP 분석 중…"):
        html_ours, err = fetch_html(url_ours)
        if err: st.error(err); st.stop()
        pack_ours = build_read_pack(html_ours)
        price_hints_ours = extract_price_hints(html_ours)

    with st.spinner("경쟁 PDP 분석 중…"):
        html_theirs, err = fetch_html(url_theirs)
        if err: st.error(err); st.stop()
        pack_theirs = build_read_pack(html_theirs)
        price_hints_theirs = extract_price_hints(html_theirs)

    # ── LLM 비교(JSON) ──
    with st.spinner("USP 비교 생성 중…"):
        prompt = make_compare_prompt(url_ours, pack_ours, price_hints_ours,
                                     url_theirs, pack_theirs, price_hints_theirs)
        schema = get_compare_schema()
        data, jerr = call_gemini_json(prompt, schema, model, thinking_off)

    if jerr or not data:
        st.error(f"LLM JSON 오류: {jerr or '결과 없음'}")
        st.code(call_gemini_text(prompt, model, thinking_off))
        st.stop()

    # 메타
    meta = data.get("meta", {}) or {}

    # ===== (요청 5,6) 포지셔닝 맵을 최상단에 표기 + 타이틀 변경 =====
    st.subheader("🧭 자사 vs. 경쟁 제품 포지셔닝 맵")
    with st.spinner("축 선정 및 점수화 중…"):
        radar_schema = get_radar_schema()
        radar_prompt = make_radar_prompt(data, meta)
        radar_json, radar_err = call_gemini_json(radar_prompt, radar_schema, model, thinking_off)

    if radar_err or not radar_json:
        st.error(f"레이더 산출 실패: {radar_err or '결과 없음'}")
        st.code(call_gemini_text(radar_prompt, model, thinking_off))
    else:
        if not HAS_MPL:
            st.warning(f"matplotlib 문제로 레이더 차트를 생략합니다. 설치 후 재실행: pip install matplotlib\n에러: {MPL_ERR}")
        else:
            axes_labels = radar_json.get("axes") or []
            ours_scores = radar_json.get("ours_scores") or []
            theirs_scores = radar_json.get("theirs_scores") or []
            ours_label = radar_json.get("ours_label") or "자사"
            theirs_label = radar_json.get("theirs_label") or "경쟁"

            n = min(len(axes_labels), len(ours_scores), len(theirs_scores))
            axes_labels = axes_labels[:n]
            ours_scores = [float(x) for x in ours_scores[:n]]
            theirs_scores = [float(x) for x in theirs_scores[:n]]

            fig, ax = radar_init(rmax=10)
            angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
            ax.set_thetagrids(np.degrees(angles), labels=axes_labels)
            ax.set_rgrids([0,2,4,6,8,10], angle=0)
            plot_radar(ax, axes_labels, ours_scores, label=ours_label)
            plot_radar(ax, axes_labels, theirs_scores, label=theirs_label)
            ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
            st.pyplot(fig)

            # (요청 1,2,3,4) 근거 섹션: 명칭 변경, 축명 대괄호 표시, 회색·소폰트, 해석 제거
            ration = radar_json.get("rationales") or []
            if ration and n:
                st.markdown(PILL_CSS, unsafe_allow_html=True)
                st.markdown("**제품 경쟁력 평가 근거**")
                for i, r in enumerate(ration[:n], 1):
                    axis_tag = axes_labels[i-1] if i-1 < len(axes_labels) else "Feature"
                    st.markdown(
                        f"<div class='small-gray'>- {i}. [{axis_tag}] {r}</div>",
                        unsafe_allow_html=True
                    )

    # ===== 요약 Pill
    st.subheader("📌 제품 USP 비교 분석")
    summary = data.get("summary", {}) or {}
    render_pills("공통", summary.get("common") or [], "common")
    render_pills("자사 제품 강점", summary.get("strength") or [], "strength")
    render_pills("자사 제품 약점", summary.get("weakness") or [], "weakness")

    # ===== 상세 테이블
    st.subheader("📊 제품 USP 비교 상세 테이블")
    rows = data.get("table", []) or []

    def enrich_price(v: str, hints: list):
        if v and ("가격" not in v and "불명확" not in v):
            return v
        if hints:
            return (v + " · 힌트: " + ", ".join(hints[:6])) if v else ("힌트: " + ", ".join(hints[:6]))
        return v or "정보 부족"

    for r in rows:
        if r.get("label","").strip() == "가격":
            r["ours"] = enrich_price(r.get("ours",""), price_hints_ours)
            r["theirs"] = enrich_price(r.get("theirs",""), price_hints_theirs)

    meta = data.get("meta", {}) or {}
    if meta:
        meta_lines = []
        if any(meta.get(k) for k in ["brand_ours","product_name_ours","product_code_ours"]):
            meta_lines.append(f"- 자사: {meta.get('brand_ours','')} {meta.get('product_name_ours','')} ({meta.get('product_code_ours','')})")
        if any(meta.get(k) for k in ["brand_theirs","product_name_theirs","product_code_theirs"]):
            meta_lines.append(f"- 경쟁: {meta.get('brand_theirs','')} {meta.get('product_name_theirs','')} ({meta.get('product_code_theirs','')})")
        if meta_lines:
            st.markdown("\n".join(meta_lines))

    df = pd.DataFrame([
        {"항목": r.get("label",""), "자사 제품": r.get("ours",""), "경쟁 제품": r.get("theirs",""), "winner": r.get("winner","")}
        for r in rows
    ])
    st.dataframe(style_winner(df), use_container_width=True)

    # ===== 소셜 카피
    st.subheader("📣 소셜 광고 카피 제안 (3개)")
    with st.spinner("카피 생성 중…"):
        ad_schema = get_adcopy_schema()
        ad_prompt = make_adcopy_prompt(data, url_ours, url_theirs)
        ad_json, ad_err = call_gemini_json(ad_prompt, ad_schema, model, thinking_off)

    if ad_err or not ad_json:
        st.error(f"카피 제안 실패: {ad_err or '결과 없음'}")
        st.code(call_gemini_text(ad_prompt, model, thinking_off))
    else:
        copies = (ad_json.get("copies") or [])[:3]
        if not copies:
            st.info("제안된 카피가 없습니다.")
        else:
            for i, c in enumerate(copies, 1):
                ch = c.get("channel") or ""
                st.markdown(f"**#{i} {('['+ch+'] ') if ch else ''}카피**")
                st.markdown(f"> {c.get('copy','')}")
                st.caption(c.get("explanation","의도 설명 없음"))

    st.success("✅ 비교 완료")
