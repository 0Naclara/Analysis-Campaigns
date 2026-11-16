import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

st.set_page_config(page_title="Simulador Preditivo de Campanhas", layout="wide")

# Carrega o estilo externo
def carregar_css(arquivo):
    with open(arquivo, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

carregar_css("style.css")

# Paleta dos gráficos Plotly
COR_ROSA_PRINCIPAL = "#f72585"
PALETA_GRAFICO = ["#f72585", "#d9048e", "#b5179e", "#7209b7", "#9d4edd", "#ff70a6"]

# 1. Carregamento dos dados
@st.cache_data
def carregar_dados():
    dados = pd.read_csv("data/campaigns_dataset.csv")
    dados["roas"] = (dados["revenue"] / dados["ad_spend"].replace(0, pd.NA)).fillna(0)
    return dados

df = carregar_dados()

# 2. Seleção automática das variáveis pré-campanha
COLUNAS_POS_CAMPANHA = {
    "impressions", "clicks", "conversions", "revenue", "profit",
    "ctr", "cpc", "cpa", "roas", "conversion_rate", "roi"
}

features_cat = []
features_num = []

for col in df.columns:
    col_lower = col.lower()
    if col_lower in COLUNAS_POS_CAMPANHA or col_lower == "id" or col_lower.endswith("_id"):
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        features_num.append(col)
    elif df[col].nunique() <= 30:
        features_cat.append(col)

TARGETS = ["conversions", "revenue"]

# 3. Treinamento do Modelo
@st.cache_resource
def treinar_modelo(dados, cols_cat, cols_num):
    X = dados[cols_cat + cols_num]
    y = dados[TARGETS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat),
            ("num", "passthrough", cols_num),
        ]
    )

    modelo = Pipeline(
        steps=[
            ("prep", preprocessador),
            ("reg", MultiOutputRegressor(
                RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
            )),
        ]
    )

    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)
    r2_conv = r2_score(y_test["conversions"], preds[:, 0])
    r2_rev = r2_score(y_test["revenue"], preds[:, 1])
    return modelo, r2_conv, r2_rev

modelo, r2_conv, r2_rev = treinar_modelo(df, features_cat, features_num)

# 4. Barra Lateral Enxuta
st.sidebar.header("Parâmetros da Campanha")
valores_usuario = {}

labels_pt = {
    "platform": "Plataforma",
    "campaign_objective": "Objetivo da Campanha",
    "audience_interest_category": "Interesse do Público",
    "creative_format": "Formato do Criativo"
}

for col, rotulo in labels_pt.items():
    if col in features_cat:
        valores_usuario[col] = st.sidebar.selectbox(rotulo, sorted(df[col].dropna().unique()))

valores_usuario["ad_spend"] = st.sidebar.slider(
    "Orçamento Investido (US$)",
    min_value=float(df["ad_spend"].min()),
    max_value=float(df["ad_spend"].max()),
    value=float(df["ad_spend"].median()),
    step=50.0,
)

for col in features_cat:
    if col not in valores_usuario:
        valores_usuario[col] = df[col].mode()[0]

for col in features_num:
    if col not in valores_usuario:
        valores_usuario[col] = float(df[col].median())

st.sidebar.markdown("---")
st.sidebar.caption(f"Precisão Receita ($R^2$): **{r2_rev:.1%}**")
st.sidebar.caption(f"Precisão Conversões ($R^2$): **{r2_conv:.1%}**")

# 5. Predição em Tempo Real
entrada_usuario = pd.DataFrame([valores_usuario])[features_cat + features_num]
predicao = modelo.predict(entrada_usuario)[0]

conv_previstas = max(0, int(round(predicao[0])))
receita_prevista = max(0.0, float(predicao[1]))
orcamento = valores_usuario["ad_spend"]
lucro_previsto = receita_prevista - orcamento
roas_previsto = receita_prevista / orcamento if orcamento > 0 else 0.0

# 6. Painel Principal
st.title("Análise Preditiva de Performance de Campanhas")
st.markdown("Ajuste os filtros na barra lateral para simular o retorno esperado antes de investir.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Conversões Previstas", f"{conv_previstas:,}")
col2.metric("Receita Estimada", f"US$ {receita_prevista:,.2f}")
col3.metric("Lucro Líquido Estimado", f"US$ {lucro_previsto:,.2f}", delta=f"{((lucro_previsto/orcamento)*100):.1f}% margem")
col4.metric("ROAS Estimado", f"{roas_previsto:.2f}x")

st.markdown("---")

# 7. Gráficos
graf1, graf2 = st.columns(2)

with graf1:
    st.subheader("Curva de Retorno por Orçamento")
    faixa_orcamento = [v for v in range(100, int(df["ad_spend"].max()) + 1, 250)]
    sim_df = pd.concat([entrada_usuario] * len(faixa_orcamento), ignore_index=True)
    sim_df["ad_spend"] = faixa_orcamento

    preds_curva = modelo.predict(sim_df)
    sim_df["Receita Prevista (US$)"] = preds_curva[:, 1]

    fig_curva = px.line(
        sim_df,
        x="ad_spend",
        y="Receita Prevista (US$)",
        labels={"ad_spend": "Orçamento (US$)"},
        title=f"Projeção para {valores_usuario.get('platform', '')} ({valores_usuario.get('creative_format', '')})",
    )
    fig_curva.update_traces(
        line=dict(color=COR_ROSA_PRINCIPAL, width=3.5),
        fill="tozeroy",
        fillcolor="rgba(247, 37, 133, 0.15)"
    )
    fig_curva.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    st.plotly_chart(fig_curva, use_container_width=True)

with graf2:
    st.subheader("ROAS Médio Histórico por Plataforma")
    df_plat = df.groupby("platform", as_index=False)["roas"].mean().sort_values("roas", ascending=False)
    fig_bar = px.bar(
        df_plat,
        x="platform",
        y="roas",
        color="platform",
        color_discrete_sequence=PALETA_GRAFICO,
        labels={"platform": "Plataforma", "roas": "ROAS Médio"},
        title="Comparativo Geral da Base Histórica",
    )
    fig_bar.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
