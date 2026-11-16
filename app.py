import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Configuração inicial da página
st.set_page_config(page_title="Simulador Preditivo de Campanhas", layout="wide")

# 1. Carregamento dos dados
@st.cache_data
def carregar_dados():
    dados = pd.read_csv("data/campaigns_dataset.csv")
    dados["roas"] = (dados["revenue"] / dados["ad_spend"].replace(0, pd.NA)).fillna(0)
    return dados

df = carregar_dados()

# Definição das colunas do modelo
FEATURES_CAT = ["platform", "campaign_objective", "audience_interest_category", "creative_format"]
FEATURE_NUM = ["ad_spend"]
TARGETS = ["conversions", "revenue"]

# Validação rápida caso alguma coluna tenha nome diferente no CSV
colunas_necessarias = FEATURES_CAT + FEATURE_NUM + TARGETS
faltantes = [c for c in colunas_necessarias if c not in df.columns]
if faltantes:
    st.error(f"Colunas não encontradas no CSV: {faltantes}")
    st.write("Colunas disponíveis no seu arquivo:", list(df.columns))
    st.stop()

# 2. Treinamento do Modelo Preditivo (fica em cache para rodar instantaneamente)
@st.cache_resource
def treinar_modelo(dados):
    X = dados[FEATURES_CAT + FEATURE_NUM]
    y = dados[TARGETS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
            ("num", "passthrough", FEATURE_NUM),
        ]
    )

    modelo = Pipeline(
        steps=[
            ("prep", preprocessador),
            ("reg", RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)),
        ]
    )

    modelo.fit(X_train, y_train)
    score_r2 = r2_score(y_test, modelo.predict(X_test))
    return modelo, score_r2

modelo, r2 = treinar_modelo(df)

# 3. Barra Lateral (Simulador de Parâmetros)
st.sidebar.header("Parâmetros da Campanha")

plataforma = st.sidebar.selectbox("Plataforma", sorted(df["platform"].unique()))
objetivo = st.sidebar.selectbox("Objetivo da Campanha", sorted(df["campaign_objective"].unique()))
interesse = st.sidebar.selectbox("Interesse do Público", sorted(df["audience_interest_category"].unique()))
formato = st.sidebar.selectbox("Formato do Criativo", sorted(df["creative_format"].unique()))

orcamento = st.sidebar.slider(
    "Orçamento Investido (US$)",
    min_value=float(df["ad_spend"].min()),
    max_value=float(df["ad_spend"].max()),
    value=float(df["ad_spend"].median()),
    step=50.0,
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Precisão do Modelo ($R^2$): **{r2:.1%}**")

# 4. Predição em Tempo Real
entrada_usuario = pd.DataFrame([{
    "platform": plataforma,
    "campaign_objective": objetivo,
    "audience_interest_category": interesse,
    "creative_format": formato,
    "ad_spend": orcamento,
}])

predicao = modelo.predict(entrada_usuario)[0]
conv_previstas = max(0, int(round(predicao[0])))
receita_prevista = max(0.0, float(predicao[1]))
lucro_previsto = receita_prevista - orcamento
roas_previsto = receita_prevista / orcamento if orcamento > 0 else 0.0

# 5. Painel Principal
st.title("Análise Preditiva de Performance de Campanhas")
st.markdown("Ajuste os filtros na barra lateral para simular o retorno esperado antes de investir.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Conversões Previstas", f"{conv_previstas:,}")
col2.metric("Receita Estimada", f"US$ {receita_prevista:,.2f}")
col3.metric("Lucro Líquido Estimado", f"US$ {lucro_previsto:,.2f}", delta=f"{((lucro_previsto/orcamento)*100):.1f}% margem")
col4.metric("ROAS Estimado", f"{roas_previsto:.2f}x")

st.markdown("---")

# 6. Gráficos de Apoio e Curva de Simulação
graf1, graf2 = st.columns(2)

with graf1:
    st.subheader("Curva de Retorno por Orçamento")
    faixa_orcamento = [v for v in range(100, int(df["ad_spend"].max()) + 1, 250)]
    sim_df = pd.DataFrame({
        "platform": [plataforma] * len(faixa_orcamento),
        "campaign_objective": [objetivo] * len(faixa_orcamento),
        "audience_interest_category": [interesse] * len(faixa_orcamento),
        "creative_format": [formato] * len(faixa_orcamento),
        "ad_spend": faixa_orcamento,
    })
    preds_curva = modelo.predict(sim_df)
    sim_df["Receita Prevista (US$)"] = preds_curva[:, 1]

    fig_curva = px.line(
        sim_df,
        x="ad_spend",
        y="Receita Prevista (US$)",
        labels={"ad_spend": "Orçamento (US$)"},
        title=f"Projeção para {plataforma} ({formato})",
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
        labels={"platform": "Plataforma", "roas": "ROAS Médio"},
        title="Comparativo Geral da Base Histórica",
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)
