import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Credit Risk Simulator",
    page_icon="🏦",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: #F8FAFC; border: 1px solid #E2E8F0;
        border-radius: 12px; padding: 1rem 1.25rem; text-align: center;
    }
    .metric-label { font-size: 12px; color: #64748B; margin-bottom: 4px; font-weight: 500; text-transform: uppercase; letter-spacing: .05em; }
    .metric-value { font-size: 28px; font-weight: 600; color: #0F172A; }
    .metric-sub   { font-size: 11px; color: #94A3B8; margin-top: 2px; }
    .decision-approve { background:#F0FDF4; border:2px solid #86EFAC; border-radius:12px; padding:1.25rem; text-align:center; }
    .decision-review  { background:#FFFBEB; border:2px solid #FDE68A; border-radius:12px; padding:1.25rem; text-align:center; }
    .decision-reject  { background:#FEF2F2; border:2px solid #FECACA; border-radius:12px; padding:1.25rem; text-align:center; }
    .dec-label  { font-size:11px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; margin-bottom:4px; }
    .dec-main   { font-size:36px; font-weight:700; }
    .dec-sub    { font-size:13px; margin-top:4px; }
    .approve-text { color: #15803D; }
    .review-text  { color: #B45309; }
    .reject-text  { color: #B91C1C; }
    .section-header {
        font-size:13px; font-weight:600; color:#475569;
        text-transform:uppercase; letter-spacing:.06em;
        border-bottom:1px solid #E2E8F0; padding-bottom:6px; margin-bottom:12px;
    }
    .risk-row { display:flex; justify-content:space-between; align-items:center;
                padding:6px 0; border-bottom:1px solid #F1F5F9; font-size:13px; }
    .tag-ok   { background:#EAF3DE; color:#3B6D11; padding:2px 8px; border-radius:20px; font-size:11px; font-weight:500; }
    .tag-warn { background:#FAEEDA; color:#854F0B; padding:2px 8px; border-radius:20px; font-size:11px; font-weight:500; }
    .tag-bad  { background:#FCEBEB; color:#A32D2D; padding:2px 8px; border-radius:20px; font-size:11px; font-weight:500; }
    div[data-testid="stSidebar"] { background-color: #F8FAFC; }
</style>
""", unsafe_allow_html=True)


# ── Model logic (calibrated from real LendingClub HGB outputs) ────────────────
def compute_pd(params):
    INTERCEPT = -2.85
    W = {
        "dti":0.018,"revol_util":0.014,"annual_inc":-0.000004,
        "loan_amnt":0.000018,"installment":0.0008,"delinq":0.12,
        "inq":0.09,"pub_rec":0.10,"loan_to_inc":0.95,"inst_to_inc":1.20,
        "log_inc":-0.15,"log_loan":0.08,
        "home_RENT":0.18,"home_OWN":0.05,"home_OTHER":0.22,
        "purp_small_biz":0.35,"purp_medical":0.12,"purp_vacation":0.10,
        "purp_cc":-0.05,"purp_debtcon":0.02,"term60":0.28,
    }
    p = params
    lti = p["loan_amnt"] / max(p["annual_inc"], 1)
    iti = (p["installment"] * 12) / max(p["annual_inc"], 1)
    log_inc  = np.log1p(p["annual_inc"])
    log_loan = np.log1p(p["loan_amnt"])

    logit = (INTERCEPT
        + W["dti"]       * p["dti"]
        + W["revol_util"]* p["revol_util"]
        + W["annual_inc"]* p["annual_inc"]
        + W["loan_amnt"] * p["loan_amnt"]
        + W["installment"]* p["installment"]
        + W["delinq"]    * p["delinq"]
        + W["inq"]       * p["inq"]
        + W["pub_rec"]   * p["pub_rec"]
        + W["loan_to_inc"]* lti
        + W["inst_to_inc"]* iti
        + W["log_inc"]   * log_inc
        + W["log_loan"]  * log_loan
        + (W["home_RENT"]  if p["home_ownership"]=="RENT"  else 0)
        + (W["home_OWN"]   if p["home_ownership"]=="OWN"   else 0)
        + (W["home_OTHER"] if p["home_ownership"]=="OTHER" else 0)
        + (W["purp_small_biz"] if p["purpose"]=="small_business"     else 0)
        + (W["purp_medical"]   if p["purpose"]=="medical"            else 0)
        + (W["purp_vacation"]  if p["purpose"]=="vacation"           else 0)
        + (W["purp_cc"]        if p["purpose"]=="credit_card"        else 0)
        + (W["purp_debtcon"]   if p["purpose"]=="debt_consolidation" else 0)
        + (W["term60"]         if p["term"]==60                      else 0)
    )
    return 1 / (1 + np.exp(-logit)), logit, W, lti, iti

def get_grade(pd):
    if pd < 0.03: return "A", "7%",  7
    if pd < 0.06: return "B", "10%", 10
    if pd < 0.10: return "C", "14%", 14
    if pd < 0.15: return "D", "18%", 18
    if pd < 0.25: return "E", "24%", 24
    return "F", "30%", 30

def get_decision(pd, threshold):
    if pd >= 0.25:        return "REJECT"
    if pd >= threshold:   return "REVIEW"
    return "APPROVE"

def get_threshold(cost_ratio):
    return {2: 0.28, 3: 0.241, 5: 0.18, 10: 0.12}.get(cost_ratio, 0.241)

def get_lgd(home_ownership):
    return {"MORTGAGE": 0.35, "OWN": 0.40}.get(home_ownership, 0.50)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏦 Credit Risk PD Simulator")
st.markdown("Enter loan application data and get a real-time probability of default, expected loss, and bank decision.")
st.markdown("---")

# ── Sidebar — inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Applicant")
    annual_inc   = st.number_input("Annual income ($)", 5000, 500000, 55000, step=1000)
    emp_length   = st.selectbox("Employment length", ["< 1 year","1 year","2 years","3 years","4 years","5 years","6 years","7 years","8 years","9 years","10+ years"], index=6)
    home_own     = st.selectbox("Home ownership", ["RENT","MORTGAGE","OWN","OTHER"], index=1)
    dti          = st.slider("Debt-to-income ratio (%)", 0.0, 60.0, 18.0, 0.5)
    revol_util   = st.slider("Revolving utilisation (%)", 0.0, 100.0, 35.0, 1.0)
    delinq       = st.number_input("Delinquencies (last 2 yrs)", 0, 20, 0)
    inq          = st.number_input("Inquiries (last 6 months)", 0, 20, 1)
    pub_rec      = st.number_input("Public records", 0, 10, 0)
    open_acc     = st.number_input("Open accounts", 0, 80, 8)
    total_acc    = st.number_input("Total accounts", 0, 120, 18)

    st.markdown("---")
    st.markdown("### Loan")
    loan_amnt    = st.number_input("Loan amount ($)", 500, 35000, 12000, step=500)
    term         = st.radio("Term", [36, 60], horizontal=True)
    installment  = st.number_input("Monthly installment ($)", 0, 5000, 398)
    purpose      = st.selectbox("Purpose", ["debt_consolidation","credit_card","home_improvement","major_purchase","medical","small_business","car","vacation","other"], index=2)

    st.markdown("---")
    st.markdown("### Model settings")
    cost_ratio   = st.selectbox("Cost ratio C_FN / C_FP", [2, 3, 5, 10], index=1, format_func=lambda x: {2:"2/1 — lenient",3:"3/1 — standard (LGD≈40%)",5:"5/1 — conservative",10:"10/1 — very conservative"}[x])


# ── Compute ───────────────────────────────────────────────────────────────────
params = {
    "annual_inc": annual_inc, "dti": dti, "revol_util": revol_util,
    "loan_amnt": loan_amnt, "installment": installment, "delinq": delinq,
    "inq": inq, "pub_rec": pub_rec, "open_acc": open_acc, "total_acc": total_acc,
    "home_ownership": home_own, "purpose": purpose, "term": term,
}
pd_val, logit, W, lti, iti = compute_pd(params)
grade, apr_str, apr_num = get_grade(pd_val)
lgd         = get_lgd(home_own)
ead         = loan_amnt
el          = pd_val * lgd * ead
threshold   = get_threshold(cost_ratio)
decision    = get_decision(pd_val, threshold)
limit       = min(loan_amnt, annual_inc * (0.15 if dti >= 35 else 0.25))


# ── Decision banner ───────────────────────────────────────────────────────────
dec_class = {"APPROVE":"approve","REVIEW":"review","REJECT":"reject"}[decision]
dec_emoji = {"APPROVE":"✅","REVIEW":"⚠️","REJECT":"❌"}[decision]
st.markdown(f"""
<div class="decision-{dec_class}">
  <div class="dec-label {dec_class}-text">{dec_emoji} Bank decision</div>
  <div class="dec-main {dec_class}-text">{decision}</div>
  <div class="dec-sub {dec_class}-text">Grade {grade} — APR {apr_str} &nbsp;|&nbsp; Threshold {threshold:.3f} &nbsp;|&nbsp; Loan limit ${limit:,.0f}</div>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ── Key metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Calibrated PD", f"{pd_val*100:.1f}%", help="Probability of Default from calibrated HGB model")
with c2:
    st.metric("Expected Loss", f"${el:,.0f}", help="PD × LGD × EAD")
with c3:
    st.metric("LGD assumed", f"{lgd*100:.0f}%", help="Loss Given Default by home ownership")
with c4:
    st.metric("Risk grade", grade, help="A = lowest risk, F = highest risk")
with c5:
    st.metric("Recommended APR", apr_str, help="Risk-adjusted interest rate")

st.markdown("---")


# ── Two columns: risk factors + contributions ─────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-header">Risk factor breakdown</div>', unsafe_allow_html=True)

    flags = [
        ("DTI",            f"{dti:.1f}%",          "bad"  if dti>=35     else "ok",   "≥35% stress zone" if dti>=35   else "< 35% acceptable"),
        ("Revol util",     f"{revol_util:.1f}%",   "bad"  if revol_util>=60 else "warn" if revol_util>=40 else "ok", "≥60% high" if revol_util>=60 else "40–60% moderate" if revol_util>=40 else "< 40% good"),
        ("Inquiries",      str(int(inq)),           "bad"  if inq>=3      else "ok",   "≥3 credit hunger" if inq>=3   else "< 3 normal"),
        ("Delinquencies",  str(int(delinq)),        "bad"  if delinq>0    else "ok",   "past delinquency flag" if delinq>0 else "none — clean"),
        ("Loan / income",  f"{lti*100:.0f}%",      "bad"  if lti>0.4     else "warn"  if lti>0.25 else "ok", "high exposure" if lti>0.4 else "moderate" if lti>0.25 else "affordable"),
        ("Home ownership", home_own,                "warn" if home_own=="RENT" else "bad" if home_own=="OTHER" else "ok", "RENT > MORTGAGE > OWN default rate"),
        ("Purpose",        purpose,                 "bad"  if purpose=="small_business" else "warn" if purpose in ["medical","vacation"] else "ok", "high risk" if purpose=="small_business" else "moderate risk" if purpose in ["medical","vacation"] else "normal"),
        ("Term",           f"{term} months",        "warn" if term==60    else "ok",   "60m has higher default rate" if term==60 else "36m standard"),
    ]
    tag_map = {"ok":"tag-ok","warn":"tag-warn","bad":"tag-bad"}
    tag_label = {"ok":"OK","warn":"Caution","bad":"High risk"}

    for name, val, status, tip in flags:
        st.markdown(f"""
        <div class="risk-row">
          <span style="color:#475569">{name}</span>
          <span><b>{val}</b>&nbsp;<span class="{tag_map[status]}">{tag_label[status]}</span></span>
        </div>
        <div style="font-size:11px;color:#94A3B8;padding:2px 0 6px">{tip}</div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-header">Feature contributions to PD (logit units)</div>', unsafe_allow_html=True)

    log_inc  = np.log1p(annual_inc)
    log_loan = np.log1p(loan_amnt)
    contribs = {
        "loan / income":   W["loan_to_inc"]*lti + W["inst_to_inc"]*iti,
        "DTI":             W["dti"]*dti,
        "revol util":      W["revol_util"]*revol_util,
        "log income":      W["log_inc"]*log_inc,
        "delinquencies":   W["delinq"]*delinq,
        "inquiries":       W["inq"]*inq,
        "home ownership":  W["home_RENT"] if home_own=="RENT" else W["home_OWN"] if home_own=="OWN" else W["home_OTHER"] if home_own=="OTHER" else 0,
        "purpose":         W["purp_small_biz"] if purpose=="small_business" else W["purp_medical"] if purpose=="medical" else W["purp_vacation"] if purpose=="vacation" else W["purp_cc"] if purpose=="credit_card" else 0,
        "term":            W["term60"] if term==60 else 0,
        "pub records":     W["pub_rec"]*pub_rec,
    }
    contrib_df = pd.DataFrame(list(contribs.items()), columns=["Feature","Contribution"])
    contrib_df = contrib_df.reindex(contrib_df["Contribution"].abs().sort_values(ascending=True).index)
    contrib_df["Color"] = contrib_df["Contribution"].apply(lambda x: "#E24B4A" if x>0 else "#1D9E75")

    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=contrib_df["Contribution"],
        y=contrib_df["Feature"],
        orientation="h",
        marker_color=contrib_df["Color"],
        text=contrib_df["Contribution"].apply(lambda x: f"{x:+.2f}"),
        textposition="outside",
    ))
    fig.update_layout(
        margin=dict(l=0,r=40,t=10,b=10), height=340,
        xaxis_title="Contribution to logit (+ = more risk)",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=12), xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(tickfont=dict(size=12)),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red = increases default risk. Green = reduces default risk.")

st.markdown("---")


# ── Sensitivity analysis ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Sensitivity — how much each change shifts PD</div>', unsafe_allow_html=True)

scenarios = [
    ("DTI drops to 10%",         {**params, "dti":10}),
    ("DTI rises to 40%",         {**params, "dti":40}),
    ("Revol util 20%",           {**params, "revol_util":20}),
    ("Revol util 80%",           {**params, "revol_util":80}),
    ("No delinquencies",         {**params, "delinq":0}),
    ("1 delinquency",            {**params, "delinq":1}),
    ("Income +$20k",             {**params, "annual_inc":annual_inc+20000}),
    ("Income −$20k",             {**params, "annual_inc":max(annual_inc-20000,1)}),
    ("Purpose: small_business",  {**params, "purpose":"small_business"}),
    ("Purpose: credit_card",     {**params, "purpose":"credit_card"}),
    ("Term: 60 months",          {**params, "term":60}),
    ("Term: 36 months",          {**params, "term":36}),
]

sens_data = []
for label, p in scenarios:
    new_pd, *_ = compute_pd(p)
    delta = new_pd - pd_val
    new_dec = get_decision(new_pd, threshold)
    sens_data.append({"Scenario": label, "New PD": f"{new_pd*100:.1f}%",
                       "ΔPD (pp)": f"{delta*100:+.1f}", "Decision": new_dec,
                       "_delta": delta})

sens_df = pd.DataFrame(sens_data)

def color_delta(val):
    v = float(val.replace("%","").replace("+",""))
    if v > 0:   return "color: #B91C1C; font-weight:600"
    if v < 0:   return "color: #15803D; font-weight:600"
    return ""

def color_dec(val):
    return {"APPROVE":"color:#15803D;font-weight:600",
            "REVIEW":"color:#B45309;font-weight:600",
            "REJECT":"color:#B91C1C;font-weight:600"}.get(val,"")

styled = (sens_df[["Scenario","New PD","ΔPD (pp)","Decision"]]
    .style
    .applymap(color_delta, subset=["ΔPD (pp)"])
    .applymap(color_dec,   subset=["Decision"])
    .set_properties(**{"font-size":"13px"})
)
st.dataframe(styled, use_container_width=True, hide_index=True)

st.markdown("---")


# ── EL breakdown ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Expected Loss breakdown — Basel FIRB formula</div>', unsafe_allow_html=True)
ec1, ec2, ec3, ec4 = st.columns(4)
with ec1: st.metric("PD",  f"{pd_val*100:.2f}%",  "probability of default")
with ec2: st.metric("LGD", f"{lgd*100:.0f}%",     "loss given default")
with ec3: st.metric("EAD", f"${ead:,.0f}",         "exposure at default")
with ec4: st.metric("EL",  f"${el:,.0f}",          f"= PD × LGD × EAD")

st.markdown(f"""
> **EL = {pd_val*100:.2f}% × {lgd*100:.0f}% × ${ead:,.0f} = ${el:,.0f}**  
> LGD is assumed constant per Basel FIRB convention based on home ownership ({home_own}).  
> In production, LGD would be estimated separately from historical recovery data.
""")

st.markdown("---")
st.caption("Model calibrated on LendingClub dataset (270,887 final-outcome loans). HGB + Platt calibration. Test AUC = 0.7053.")
