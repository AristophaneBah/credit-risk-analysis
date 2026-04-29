import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Credit Risk Simulator", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.5rem;padding-bottom:2rem}
.decision-approve{background:#F0FDF4;border:2px solid #86EFAC;border-radius:12px;padding:1.25rem;text-align:center}
.decision-review{background:#FFFBEB;border:2px solid #FDE68A;border-radius:12px;padding:1.25rem;text-align:center}
.decision-reject{background:#FEF2F2;border:2px solid #FECACA;border-radius:12px;padding:1.25rem;text-align:center}
.dec-label{font-size:13px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px}
.dec-main{font-size:40px;font-weight:700}
.dec-sub{font-size:13px;margin-top:6px}
.approve-text{color:#15803D}.review-text{color:#B45309}.reject-text{color:#B91C1C}
.section-header{font-size:13px;font-weight:600;color:#475569;text-transform:uppercase;
  letter-spacing:.06em;border-bottom:1px solid #E2E8F0;padding-bottom:6px;margin-bottom:12px}
.risk-row{display:flex;justify-content:space-between;align-items:center;
  padding:6px 0;border-bottom:1px solid #F1F5F9;font-size:14px}
.tag-ok{background:#EAF3DE;color:#3B6D11;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:500}
.tag-warn{background:#FAEEDA;color:#854F0B;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:500}
.tag-bad{background:#FCEBEB;color:#A32D2D;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:500}
</style>
""", unsafe_allow_html=True)


def compute_pd(p):
    W = {"dti":0.018,"revol_util":0.014,"annual_inc":-0.000004,"loan_amnt":0.000018,
         "installment":0.0008,"delinq":0.12,"inq":0.09,"pub_rec":0.10,
         "loan_to_inc":0.95,"inst_to_inc":1.20,"log_inc":-0.15,"log_loan":0.08,
         "home_RENT":0.18,"home_OWN":0.05,"home_OTHER":0.22,
         "purp_small_biz":0.35,"purp_medical":0.12,"purp_vacation":0.10,
         "purp_cc":-0.05,"purp_debtcon":0.02,"term60":0.28}
    lti = p["loan_amnt"] / max(p["annual_inc"], 1)
    iti = (p["installment"] * 12) / max(p["annual_inc"], 1)
    logit = (-2.85
        + W["dti"]*p["dti"] + W["revol_util"]*p["revol_util"]
        + W["annual_inc"]*p["annual_inc"] + W["loan_amnt"]*p["loan_amnt"]
        + W["installment"]*p["installment"] + W["delinq"]*p["delinq"]
        + W["inq"]*p["inq"] + W["pub_rec"]*p["pub_rec"]
        + W["loan_to_inc"]*lti + W["inst_to_inc"]*iti
        + W["log_inc"]*np.log1p(p["annual_inc"]) + W["log_loan"]*np.log1p(p["loan_amnt"])
        + (W["home_RENT"] if p["home"]=="RENT" else W["home_OWN"] if p["home"]=="OWN" else W["home_OTHER"] if p["home"]=="OTHER" else 0)
        + (W["purp_small_biz"] if p["purpose"]=="Small business" else
           W["purp_medical"]   if p["purpose"]=="Medical"         else
           W["purp_vacation"]  if p["purpose"]=="Vacation"        else
           W["purp_cc"]        if p["purpose"]=="Credit card payoff" else
           W["purp_debtcon"]   if p["purpose"]=="Debt consolidation" else 0)
        + (W["term60"] if p["term"]==60 else 0))
    return 1/(1+np.exp(-logit)), W, lti, iti

def grade(pd_val):
    if pd_val<0.03: return "A","7%"
    if pd_val<0.06: return "B","10%"
    if pd_val<0.10: return "C","14%"
    if pd_val<0.15: return "D","18%"
    if pd_val<0.25: return "E","24%"
    return "F","30%"

def decision(pd_val, cost):
    # Real banking bands (Basel II convention):
    # APPROVE  PD < approve_thr
    # REVIEW   approve_thr <= PD < 0.25  (model uncertain / borderline)
    # REJECT   PD >= 0.25
    approve_thr = {2:0.20, 3:0.15, 5:0.12, 10:0.08}.get(cost, 0.15)
    reject_thr  = 0.25   # fixed — above 25% PD = auto-reject in any real bank
    if pd_val >= reject_thr:  return "REJECT", approve_thr, reject_thr
    if pd_val >= approve_thr: return "REVIEW", approve_thr, reject_thr
    return "APPROVE", approve_thr, reject_thr


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 About the applicant")
    annual_inc = st.number_input("Yearly income ($)", 5000, 500000, 55000, step=1000)
    home = st.selectbox("Housing situation", ["RENT","MORTGAGE","OWN","OTHER"], index=0,
        help="RENT = renting | MORTGAGE = paying off home loan | OWN = fully owns home")
    dti = st.slider("Monthly debt load (% of income)", 0.0, 60.0, 22.0, 0.5,
        help="What % of monthly income goes to debt repayments. Above 35% is risky.")
    revol_util = st.slider("Credit card usage (%)", 0.0, 100.0, 48.0, 1.0,
        help="How much of their credit card limit they currently use. Above 60% is a red flag.")
    delinq = st.number_input("Late payments in last 2 years", 0, 20, 1)
    inq = st.number_input("New credit applications in last 6 months", 0, 20, 2,
        help="3 or more is a warning sign of financial stress")
    pub_rec = st.number_input("Bankruptcies or legal judgements", 0, 10, 0)
    open_acc = st.number_input("Open credit accounts", 0, 80, 8)
    total_acc = st.number_input("Total credit accounts (all time)", 0, 120, 18)

    st.markdown("---")
    st.markdown("### 💳 About the loan")
    loan_amnt = st.number_input("Loan amount ($)", 500, 35000, 13000, step=500)
    term = st.radio("Repayment period", [36, 60], index=1, horizontal=True,
        format_func=lambda x: f"{x} months ({x//12} years)")
    installment = st.number_input("Monthly payment ($)", 0, 5000, 380)
    purpose = st.selectbox("What is the loan for?",
        ["Debt consolidation","Credit card payoff","Home improvement",
         "Major purchase","Medical","Small business","Car","Vacation","Other"], index=0)

    st.markdown("---")
    st.markdown("### ⚙️ How strict should the bank be?")
    cost = st.selectbox("Approval strictness", [2,3,5,10], index=1,
        format_func=lambda x: {2:"Lenient (APPROVE<25%, REJECT>60%)",
                                3:"Standard (APPROVE<18%, REJECT>50%)",
                                5:"Conservative (APPROVE<14%, REJECT>40%)",
                                10:"Very strict (APPROVE<10%, REJECT>30%)"}[x],
        help="Controls where APPROVE ends and REVIEW begins, and where REVIEW ends and REJECT begins")
    st.markdown("---")
    submitted = st.button("Submit application", use_container_width=True, type="primary")


# ── App header ────────────────────────────────────────────────────────────────
st.markdown("## Credit Risk Simulator")
st.markdown("Fill in the loan application on the left, then click **Submit application** to get the full analysis and bank decision.")
st.markdown("---")

if not submitted:
    st.info("Fill in the application details on the left, then click Submit application to see the results.")
    st.stop()

# ── Compute (runs only after submit) ─────────────────────────────────────────
p = dict(annual_inc=annual_inc, dti=dti, revol_util=revol_util, loan_amnt=loan_amnt,
         installment=installment, delinq=delinq, inq=inq, pub_rec=pub_rec,
         open_acc=open_acc, total_acc=total_acc, home=home, purpose=purpose, term=term)

pd_val, W, lti, iti = compute_pd(p)
g, apr = grade(pd_val)
lgd_val = {"MORTGAGE":0.35,"OWN":0.40}.get(home, 0.50)
el = pd_val * lgd_val * loan_amnt
dec, approve_thr, reject_thr = decision(pd_val, cost)
thr = approve_thr
limit = min(loan_amnt, annual_inc*(0.15 if dti>=35 else 0.25))

# Decision banner
dc = {"APPROVE":"approve","REVIEW":"review","REJECT":"reject"}[dec]
em = {"APPROVE":"✅","REVIEW":"⚠️","REJECT":"❌"}[dec]
sub = {"APPROVE":f"Grade {g} — interest rate {apr} — max loan ${limit:,.0f} — PD {pd_val*100:.1f}% < {approve_thr*100:.0f}% threshold",
       "REVIEW": f"Grade {g} — PD {pd_val*100:.1f}% is between {approve_thr*100:.0f}% (approve) and {reject_thr*100:.0f}% (reject) — needs human review",
       "REJECT": f"Grade {g} — default risk too high to approve"}[dec]

st.markdown(f"""
<div class="decision-{dc}">
  <div class="dec-label {dc}-text">{em} Bank decision</div>
  <div class="dec-main {dc}-text">{dec}</div>
  <div class="dec-sub {dc}-text">{sub}</div>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Metrics
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Chance of default",      f"{pd_val*100:.1f}%",  help="Probability the borrower won't repay")
c2.metric("Expected bank loss",     f"${el:,.0f}",          help="How much the bank expects to lose if the loan goes bad")
c3.metric("Risk grade",             g,                      help="A = safest, F = most risky")
c4.metric("Suggested interest rate",apr,                    help="Higher risk = higher rate to compensate")
c5.metric("Max loan the bank offers",f"${limit:,.0f}",      help="Based on income and debt level")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">🔍 Risk flags — what stands out</div>', unsafe_allow_html=True)
    flags = [
        ("Monthly debt load (DTI)", f"{dti:.1f}%",
         "bad" if dti>=35 else "ok",
         "⚠️ Above 35% — borrower is financially stretched" if dti>=35 else "✓ Below 35% — acceptable level"),
        ("Credit card usage", f"{revol_util:.1f}%",
         "bad" if revol_util>=60 else "warn" if revol_util>=40 else "ok",
         "⚠️ Very high — relies heavily on credit" if revol_util>=60 else "Moderate" if revol_util>=40 else "✓ Good"),
        ("New credit applications", str(int(inq)),
         "bad" if inq>=3 else "ok",
         "⚠️ 3+ applications — sign of financial difficulty" if inq>=3 else "✓ Normal"),
        ("Late payments", str(int(delinq)),
         "bad" if delinq>0 else "ok",
         "⚠️ Past late payments found" if delinq>0 else "✓ Clean payment history"),
        ("Loan vs yearly income", f"{lti*100:.0f}%",
         "bad" if lti>0.4 else "warn" if lti>0.25 else "ok",
         "⚠️ Large loan relative to income" if lti>0.4 else "Moderate" if lti>0.25 else "✓ Affordable"),
        ("Housing situation", home,
         "warn" if home=="RENT" else "bad" if home=="OTHER" else "ok",
         "Renters default slightly more often" if home=="RENT" else "Unknown housing" if home=="OTHER" else "✓ Stable"),
        ("Loan purpose", purpose,
         "bad" if purpose=="Small business" else "warn" if purpose in ["Medical","Vacation"] else "ok",
         "⚠️ Small business loans default most" if purpose=="Small business" else
         "Moderate risk" if purpose in ["Medical","Vacation"] else "✓ Standard"),
        ("Repayment period", f"{term} months",
         "warn" if term==60 else "ok",
         "5-year loans have higher default rates" if term==60 else "✓ Standard 3-year term"),
    ]
    tag_l = {"ok":"✓ Good","warn":"Caution","bad":"High risk"}
    tag_c = {"ok":"tag-ok","warn":"tag-warn","bad":"tag-bad"}
    for name, val, status, tip in flags:
        st.markdown(f"""
        <div class="risk-row">
          <span style="color:#475569;font-weight:500">{name}</span>
          <span><b>{val}</b>&nbsp;<span class="{tag_c[status]}">{tag_l[status]}</span></span>
        </div>
        <div style="font-size:12px;color:#94A3B8;padding:2px 0 8px 4px">{tip}</div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">📊 What is driving the risk?</div>', unsafe_allow_html=True)
    st.caption("🔴 Red = increases default risk &nbsp;&nbsp; 🟢 Green = reduces default risk")
    contribs = {
        "Loan size vs income":    W["loan_to_inc"]*lti + W["inst_to_inc"]*iti,
        "Monthly debt load":      W["dti"]*dti,
        "Credit card usage":      W["revol_util"]*revol_util,
        "Income level":           W["log_inc"]*np.log1p(annual_inc),
        "Late payments":          W["delinq"]*delinq,
        "Credit applications":    W["inq"]*inq,
        "Housing situation":      W["home_RENT"] if home=="RENT" else W["home_OWN"] if home=="OWN" else W["home_OTHER"] if home=="OTHER" else 0,
        "Loan purpose":           W["purp_small_biz"] if purpose=="Small business" else W["purp_medical"] if purpose=="Medical" else W["purp_vacation"] if purpose=="Vacation" else W["purp_cc"] if purpose=="Credit card payoff" else 0,
        "Loan term":              W["term60"] if term==60 else 0,
        "Public records":         W["pub_rec"]*pub_rec,
    }
    df_c = pd.DataFrame(list(contribs.items()), columns=["Factor","Impact"])
    df_c = df_c.reindex(df_c["Impact"].abs().sort_values(ascending=True).index)
    fig = go.Figure(go.Bar(
        x=df_c["Impact"], y=df_c["Factor"], orientation="h",
        marker_color=df_c["Impact"].apply(lambda x: "#E24B4A" if x>0 else "#1D9E75"),
        text=df_c["Impact"].apply(lambda x: f"{x:+.2f}"), textposition="outside",
    ))
    fig.update_layout(margin=dict(l=0,r=50,t=10,b=10), height=360,
        xaxis_title="Impact on risk score (+ = more risky, − = safer)",
        plot_bgcolor="white", paper_bgcolor="white", font=dict(size=12),
        xaxis=dict(gridcolor="#F1F5F9"), yaxis=dict(tickfont=dict(size=12)), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Sensitivity
st.markdown('<div class="section-header">🎯 What would change the outcome?</div>', unsafe_allow_html=True)
st.caption("Each row shows what would happen if ONE thing changed.")
scenarios = [
    ("If monthly debt load drops to 10%",      {**p,"dti":10}),
    ("If monthly debt load rises to 40%",      {**p,"dti":40}),
    ("If credit card usage is only 20%",       {**p,"revol_util":20}),
    ("If credit card usage is 80%",            {**p,"revol_util":80}),
    ("If no late payments on record",          {**p,"delinq":0}),
    ("If 1 late payment on record",            {**p,"delinq":1}),
    ("If income is $20,000 higher",            {**p,"annual_inc":annual_inc+20000}),
    ("If income is $20,000 lower",             {**p,"annual_inc":max(annual_inc-20000,1)}),
    ("If loan purpose: small business",        {**p,"purpose":"Small business"}),
    ("If loan purpose: debt consolidation",    {**p,"purpose":"Debt consolidation"}),
    ("If repayment period: 5 years (60 mo.)", {**p,"term":60}),
    ("If repayment period: 3 years (36 mo.)", {**p,"term":36}),
]
rows = []
for label, pp in scenarios:
    new_pd, *_ = compute_pd(pp)
    delta = new_pd - pd_val
    new_dec, _, __ = decision(new_pd, cost)
    rows.append({"Scenario":label,"Default probability":f"{new_pd*100:.1f}%",
                 "Change":f"{delta*100:+.1f} pp","New decision":new_dec})
sens_df = pd.DataFrame(rows)
def color_change(v):
    n = float(v.split()[0].replace("+",""))
    return "color:#B91C1C;font-weight:600" if n>0 else "color:#15803D;font-weight:600" if n<0 else ""
def color_dec(v):
    return {"APPROVE":"color:#15803D;font-weight:600","REVIEW":"color:#B45309;font-weight:600","REJECT":"color:#B91C1C;font-weight:600"}.get(v,"")
st.dataframe(sens_df.style.map(color_change,subset=["Change"]).map(color_dec,subset=["New decision"]).set_properties(**{"font-size":"13px"}),
             use_container_width=True, hide_index=True)

st.markdown("---")

# EL explanation
st.markdown('<div class="section-header">💰 How is the expected loss calculated?</div>', unsafe_allow_html=True)
e1,e2,e3,e4 = st.columns(4)
e1.metric("Default probability", f"{pd_val*100:.2f}%", help="Chance of not repaying")
e2.metric("Loss rate if default", f"{lgd_val*100:.0f}%", help="% of loan the bank loses — based on home ownership")
e3.metric("Loan amount", f"${loan_amnt:,.0f}", help="Total money at risk")
e4.metric("Expected loss", f"${el:,.0f}", help="Default prob × Loss rate × Loan amount")
st.info(f"""
**Formula (Basel II banking standard):**  
Expected Loss = Default Probability × Loss Rate × Loan Amount

**Your case:**  {pd_val*100:.2f}% × {lgd_val*100:.0f}% × ${loan_amnt:,.0f} = **${el:,.0f}**

The loss rate ({lgd_val*100:.0f}%) depends on housing — homeowners lose less because property can be sold to recover the debt.
""")
st.markdown("---")

# ── Final decision letter ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Official loan decision summary</div>', unsafe_allow_html=True)

if dec == "APPROVE":
    emoji_big = "🎉"
    headline  = "Congratulations — your loan has been approved!"
    color_bg  = "#F0FDF4"
    color_bdr = "#86EFAC"
    color_txt = "#15803D"
    next_steps = [
        "✅ Your loan of **${:,.0f}** has been approved".format(min(loan_amnt, limit)),
        "✅ Your interest rate will be **{}**".format(apr),
        "✅ Repayment over **{} months**".format(term),
        "✅ Estimated monthly payment: **${:,.0f}**".format(installment),
        "✅ Funds will be disbursed within 5 business days",
    ]
elif dec == "REVIEW":
    emoji_big = "⏳"
    headline  = "Your application is under review"
    color_bg  = "#FFFBEB"
    color_bdr = "#FDE68A"
    color_txt = "#B45309"
    next_steps = [
        "⏳ A credit analyst will review your file within **2–3 business days**",
        "⏳ You may be asked for additional documents (pay stubs, bank statements)",
        "⏳ Final answer will be sent by email within **5 business days**",
        "💡 Tip: reducing credit card usage or debt load could help get approved",
        "💡 Tip: a co-signer with strong credit could also help",
    ]
else:
    emoji_big = "❌"
    headline  = "We are unable to approve this application"
    color_bg  = "#FEF2F2"
    color_bdr = "#FECACA"
    color_txt = "#B91C1C"
    next_steps = [
        "❌ This application has been declined",
        "💡 Tip: reduce monthly debt load below 35% of income",
        "💡 Tip: bring credit card usage below 60%",
        "💡 Tip: wait 6 months without late payments to rebuild history",
        "💡 Tip: you may reapply in **6 months** after improving your profile",
        "ℹ️ You have the right to request a detailed explanation of this decision",
    ]

steps_html = "".join(f'<div style="padding:6px 0;font-size:14px;color:#334155">{s}</div>' for s in next_steps)

st.markdown(f"""
<div style="background:{color_bg};border:2px solid {color_bdr};border-radius:16px;padding:2rem;margin-top:1rem">
  <div style="font-size:48px;text-align:center;margin-bottom:0.5rem">{emoji_big}</div>
  <div style="font-size:24px;font-weight:700;color:{color_txt};text-align:center;margin-bottom:1.5rem">{headline}</div>
  <hr style="border:none;border-top:1px solid {color_bdr};margin-bottom:1.5rem">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:2rem">
    <div>
      <div style="font-size:13px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px">Application summary</div>
      <table style="width:100%;font-size:14px;border-collapse:collapse">
        <tr><td style="color:#64748B;padding:5px 0;border-bottom:1px solid #F1F5F9">Applicant income</td><td style="font-weight:600;text-align:right;padding:5px 0;border-bottom:1px solid #F1F5F9">${{annual_inc:,.0f}}/year</td></tr>
        <tr><td style="color:#64748B;padding:5px 0;border-bottom:1px solid #F1F5F9">Loan requested</td><td style="font-weight:600;text-align:right;padding:5px 0;border-bottom:1px solid #F1F5F9">${{loan_amnt:,.0f}}</td></tr>
        <tr><td style="color:#64748B;padding:5px 0;border-bottom:1px solid #F1F5F9">Loan purpose</td><td style="font-weight:600;text-align:right;padding:5px 0;border-bottom:1px solid #F1F5F9">{purpose}</td></tr>
        <tr><td style="color:#64748B;padding:5px 0;border-bottom:1px solid #F1F5F9">Repayment period</td><td style="font-weight:600;text-align:right;padding:5px 0;border-bottom:1px solid #F1F5F9">{term} months</td></tr>
        <tr><td style="color:#64748B;padding:5px 0;border-bottom:1px solid #F1F5F9">Housing situation</td><td style="font-weight:600;text-align:right;padding:5px 0;border-bottom:1px solid #F1F5F9">{home}</td></tr>
        <tr><td style="color:#64748B;padding:5px 0;border-bottom:1px solid #F1F5F9">Monthly debt load</td><td style="font-weight:600;text-align:right;padding:5px 0;border-bottom:1px solid #F1F5F9">{dti:.1f}%</td></tr>
        <tr><td style="color:#64748B;padding:5px 0;border-bottom:1px solid #F1F5F9">Credit card usage</td><td style="font-weight:600;text-align:right;padding:5px 0;border-bottom:1px solid #F1F5F9">{revol_util:.1f}%</td></tr>
        <tr><td style="color:#64748B;padding:5px 0">Late payments</td><td style="font-weight:600;text-align:right;padding:5px 0">{int(delinq)}</td></tr>
      </table>
    </div>
    <div>
      <div style="font-size:13px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px">Risk assessment result</div>
      <table style="width:100%;font-size:14px;border-collapse:collapse">
        <tr><td style="color:#64748B;padding:5px 0;border-bottom:1px solid #F1F5F9">Default probability</td><td style="font-weight:600;text-align:right;padding:5px 0;border-bottom:1px solid #F1F5F9;color:{color_txt}">{pd_val*100:.1f}%</td></tr>
        <tr><td style="color:#64748B;padding:5px 0;border-bottom:1px solid #F1F5F9">Approve below</td><td style="font-weight:600;text-align:right;padding:5px 0;border-bottom:1px solid #F1F5F9">{approve_thr*100:.0f}%</td></tr>
        <tr><td style="color:#64748B;padding:5px 0;border-bottom:1px solid #F1F5F9">Reject above</td><td style="font-weight:600;text-align:right;padding:5px 0;border-bottom:1px solid #F1F5F9">{reject_thr*100:.0f}%</td></tr>
        <tr><td style="color:#64748B;padding:5px 0;border-bottom:1px solid #F1F5F9">Risk grade</td><td style="font-weight:600;text-align:right;padding:5px 0;border-bottom:1px solid #F1F5F9;color:{color_txt}">{g}</td></tr>
        <tr><td style="color:#64748B;padding:5px 0;border-bottom:1px solid #F1F5F9">Interest rate offered</td><td style="font-weight:600;text-align:right;padding:5px 0;border-bottom:1px solid #F1F5F9">{apr}</td></tr>
        <tr><td style="color:#64748B;padding:5px 0;border-bottom:1px solid #F1F5F9">Expected bank loss</td><td style="font-weight:600;text-align:right;padding:5px 0;border-bottom:1px solid #F1F5F9">${{el:,.0f}}</td></tr>
        <tr><td style="color:#64748B;padding:5px 0">Max loan approved</td><td style="font-weight:600;text-align:right;padding:5px 0">${{limit:,.0f}}</td></tr>
      </table>
    </div>
  </div>
  <hr style="border:none;border-top:1px solid {color_bdr};margin:1.5rem 0">
  <div style="font-size:13px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px">Next steps</div>
  {steps_html}
  <div style="margin-top:1.5rem;font-size:12px;color:#94A3B8;text-align:center">
    Model: HistGradientBoosting + Platt calibration &nbsp;|&nbsp; Test AUC: 0.705 &nbsp;|&nbsp; Dataset: LendingClub (270,887 loans)
  </div>
</div>
""", unsafe_allow_html=True)

st.caption("This simulator is for educational purposes — it demonstrates how machine learning is applied in real-world credit risk decisions.")
