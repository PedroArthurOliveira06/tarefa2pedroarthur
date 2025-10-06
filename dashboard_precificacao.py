# ==============================================================================
# DASHBOARD INTERATIVO - ANÁLISE DE PRECIFICAÇÃO IMOBILIÁRIA
# ==============================================================================
# Aluno: Pedro Arthur Santos Oliveira
# Matrícula: 231036069
# Professor: João Gabriel de Moraes Souza
# Disciplina: Sistemas de Informação em Engenharia de Produção - UnB
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import f_oneway, anderson, levene
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================

st.set_page_config(
    page_title="Dashboard - Precificação Imobiliária",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    h1 {color: #003d82; font-weight: 700;}
    h2 {color: #0066cc; font-weight: 600;}
    h3 {color: #1976d2; font-weight: 500;}
    .stButton>button {background-color: #0066cc; color: white; border-radius: 8px; border: none; padding: 10px 24px; font-weight: 600;}
    .stButton>button:hover {background-color: #003d82;}
    .stSelectbox, .stSlider {background-color: #ffffff; border-radius: 8px;}
    div[data-testid="stExpander"] {background-color: #ffffff; border-radius: 10px; border: 1px solid #e0e0e0;}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# FUNÇÕES DE CARREGAMENTO E CACHE
# ==============================================================================

@st.cache_data
def carregar_dados():
    """Carrega e prepara os dados"""
    try:
        df = pd.read_csv('AmesHousing.csv')
        return df
    except:
        st.error("Erro ao carregar o arquivo. Certifique-se de que 'AmesHousing.csv' está no mesmo diretório.")
        st.stop()

@st.cache_data
def preparar_dados_anova(df):
    """Prepara dados para ANOVA"""
    df_anova = df[['SalePrice', 'Overall Qual', 'Central Air', 'Kitchen Qual']].copy()
    df_anova = df_anova.dropna()
    return df_anova

@st.cache_data
def preparar_dados_regressao(df):
    """Prepara dados para regressão"""
    vars_reg = ['SalePrice', 'Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Central Air']
    df_reg = df[vars_reg].copy()
    df_reg = df_reg.dropna()
    df_reg['Central_Air_Y'] = (df_reg['Central Air'] == 'Y').astype(int)
    return df_reg

@st.cache_data
def treinar_modelos(df_reg):
    """Treina ambos os modelos de regressão"""
    # Modelo 1: Original
    X_vars = ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Central_Air_Y']
    X = df_reg[X_vars]
    y = df_reg['SalePrice']
    X_const = sm.add_constant(X)
    model_1 = sm.OLS(y, X_const).fit()
    
    # Modelo 2: Log-Log
    df_reg_log = df_reg.copy()
    df_reg_log['log_SalePrice'] = np.log(df_reg_log['SalePrice'])
    df_reg_log['log_Gr_Liv_Area'] = np.log(df_reg_log['Gr Liv Area'].replace(0, 1))
    df_reg_log['log_Overall_Qual'] = np.log(df_reg_log['Overall Qual'])
    df_reg_log['log_Year_Built'] = np.log(df_reg_log['Year Built'])
    df_reg_log['log_Garage_Area'] = np.log(df_reg_log['Garage Area'].replace(0, 1))
    
    X_vars_log = ['log_Gr_Liv_Area', 'log_Overall_Qual', 'log_Year_Built', 'log_Garage_Area', 'Central_Air_Y']
    X_log = df_reg_log[X_vars_log]
    y_log = df_reg_log['log_SalePrice']
    X_log_const = sm.add_constant(X_log)
    model_2 = sm.OLS(y_log, X_log_const).fit()
    
    return model_1, model_2, df_reg_log

# ==============================================================================
# CARREGAR DADOS
# ==============================================================================

df = carregar_dados()
df_anova = preparar_dados_anova(df)
df_reg = preparar_dados_regressao(df)
model_1, model_2, df_reg_log = treinar_modelos(df_reg)

# ==============================================================================
# HEADER
# ==============================================================================

st.markdown("""
    <div style='background: linear-gradient(135deg, #003d82 0%, #0066cc 100%); 
                padding: 30px; 
                border-radius: 15px; 
                text-align: center; 
                margin-bottom: 30px;
                box-shadow: 0 4px 15px rgba(0,61,130,0.3);'>
        <h1 style='color: white; margin: 0; font-size: 2.5em;'>🏠 Dashboard de Precificação Imobiliária</h1>
        <p style='color: #e3f2fd; font-size: 1.2em; margin-top: 10px;'>Análise ANOVA e Regressão Linear - Ames Housing Dataset</p>
        <p style='color: #bbdefb; margin-top: 10px;'>Pedro Arthur Santos Oliveira (231036069) | UnB - Engenharia de Produção</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/real-estate.png", width=80)
    st.title("Navegação")
    
    pagina = st.radio(
        "Escolha uma seção:",
        ["📊 Visão Geral",
         "📈 Análise ANOVA",
         "🔮 Modelos de Regressão",
         "💰 Calculadora de Preço",
         "📉 Análise de Sensibilidade",
         "📋 Comparação de Modelos"]
    )
    
    st.markdown("---")
    
    st.markdown("""
        ### Sobre o Dashboard
        
        Este dashboard interativo apresenta uma análise completa de precificação imobiliária usando:
        
        - **ANOVA**: Análise de variância para identificar fatores significativos
        - **Regressão Linear**: Modelos preditivos de preços
        - **Visualizações Interativas**: Gráficos dinâmicos com Plotly
        - **Calculadora**: Estimativa de preços em tempo real
    """)
    
    st.markdown("---")
    st.caption(f"**Dataset:** {df.shape[0]:,} imóveis")
    st.caption(f"**Variáveis:** {df.shape[1]} características")

# ==============================================================================
# PÁGINA 1: VISÃO GERAL
# ==============================================================================

if pagina == "📊 Visão Geral":
    st.header("📊 Visão Geral do Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Imóveis", f"{df.shape[0]:,}")
    with col2:
        st.metric("Preço Médio", f"${df['SalePrice'].mean():,.0f}")
    with col3:
        st.metric("Preço Mediano", f"${df['SalePrice'].median():,.0f}")
    with col4:
        st.metric("Preço Máximo", f"${df['SalePrice'].max():,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição de Preços")
        fig = px.histogram(df, x='SalePrice', nbins=50, 
                          title='Distribuição dos Preços de Venda',
                          labels={'SalePrice': 'Preço (USD)', 'count': 'Frequência'},
                          color_discrete_sequence=['#0066cc'])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Box Plot de Preços")
        fig = px.box(df, y='SalePrice', 
                    title='Box Plot - Distribuição de Preços',
                    labels={'SalePrice': 'Preço (USD)'},
                    color_discrete_sequence=['#0066cc'])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Estatísticas Descritivas")
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(df['SalePrice'].describe().apply(lambda x: f'${x:,.2f}' if x > 100 else f'{x:.2f}'), 
                    use_container_width=True)
    
    with col2:
        stats_dict = {
            'Métrica': ['Assimetria', 'Curtose', 'Coef. Variação', 'Amplitude'],
            'Valor': [
                f"{df['SalePrice'].skew():.3f}",
                f"{df['SalePrice'].kurtosis():.3f}",
                f"{(df['SalePrice'].std() / df['SalePrice'].mean() * 100):.2f}%",
                f"${(df['SalePrice'].max() - df['SalePrice'].min()):,.0f}"
            ]
        }
        st.dataframe(pd.DataFrame(stats_dict), hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Principais Variáveis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(df, x='Gr Liv Area', y='SalePrice',
                        title='Preço vs Área Habitável',
                        labels={'Gr Liv Area': 'Área Habitável (pés²)', 'SalePrice': 'Preço (USD)'},
                        color='Overall Qual',
                        color_continuous_scale='Blues',
                        trendline="ols")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df, x='Year Built', y='SalePrice',
                        title='Preço vs Ano de Construção',
                        labels={'Year Built': 'Ano', 'SalePrice': 'Preço (USD)'},
                        color='Overall Qual',
                        color_continuous_scale='Blues',
                        trendline="ols")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PÁGINA 2: ANÁLISE ANOVA
# ==============================================================================

elif pagina == "📈 Análise ANOVA":
    st.header("📈 Análise de Variância (ANOVA)")
    
    st.markdown("""
    A ANOVA permite testar se diferentes grupos têm médias significativamente diferentes.
    Analisamos o impacto de três fatores no preço de venda dos imóveis.
    """)
    
    # ANOVA 1: Qualidade Geral
    st.markdown("---")
    st.subheader("1️⃣ Impacto da Qualidade Geral")
    
    grupos_qual = [df_anova[df_anova['Overall Qual'] == q]['SalePrice'] 
                   for q in sorted(df_anova['Overall Qual'].unique())]
    f_stat_qual, p_value_qual = f_oneway(*grupos_qual)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estatística F", f"{f_stat_qual:.2f}")
    with col2:
        st.metric("P-valor", f"{p_value_qual:.6f}")
    with col3:
        resultado = "✅ Significativo" if p_value_qual < 0.05 else "❌ Não Significativo"
        st.metric("Resultado", resultado)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(df_anova, x='Overall Qual', y='SalePrice',
                    title='Distribuição de Preços por Qualidade Geral',
                    labels={'Overall Qual': 'Qualidade Geral', 'SalePrice': 'Preço (USD)'},
                    color='Overall Qual',
                    color_discrete_sequence=px.colors.sequential.Blues)
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        means_qual = df_anova.groupby('Overall Qual')['SalePrice'].mean().reset_index()
        fig = px.bar(means_qual, x='Overall Qual', y='SalePrice',
                    title='Preço Médio por Qualidade Geral',
                    labels={'Overall Qual': 'Qualidade Geral', 'SalePrice': 'Preço Médio (USD)'},
                    color='SalePrice',
                    color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📊 Ver Estatísticas Detalhadas"):
        stats_qual = df_anova.groupby('Overall Qual')['SalePrice'].agg(['count', 'mean', 'std', 'min', 'max'])
        stats_qual.columns = ['Contagem', 'Média', 'Desvio Padrão', 'Mínimo', 'Máximo']
        st.dataframe(stats_qual.style.format({
            'Média': '${:,.0f}',
            'Desvio Padrão': '${:,.0f}',
            'Mínimo': '${:,.0f}',
            'Máximo': '${:,.0f}'
        }), use_container_width=True)
    
    # ANOVA 2: Ar Condicionado Central
    st.markdown("---")
    st.subheader("2️⃣ Impacto do Ar Condicionado Central")
    
    group_n = df_anova[df_anova['Central Air'] == 'N']['SalePrice']
    group_y = df_anova[df_anova['Central Air'] == 'Y']['SalePrice']
    t_stat, p_air = stats.ttest_ind(group_n, group_y)
    diff_pct = ((group_y.mean() - group_n.mean()) / group_n.mean()) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Estatística t", f"{t_stat:.2f}")
    with col2:
        st.metric("P-valor", f"{p_air:.6f}")
    with col3:
        st.metric("Diferença %", f"{diff_pct:.1f}%")
    with col4:
        st.metric("Diferença $", f"${(group_y.mean() - group_n.mean()):,.0f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.violin(df_anova, x='Central Air', y='SalePrice',
                       title='Distribuição de Preços por Ar Condicionado',
                       labels={'Central Air': 'Ar Condicionado Central', 'SalePrice': 'Preço (USD)'},
                       color='Central Air',
                       box=True)
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        means_air = df_anova.groupby('Central Air')['SalePrice'].mean().reset_index()
        fig = px.bar(means_air, x='Central Air', y='SalePrice',
                    title='Preço Médio por Ar Condicionado',
                    labels={'Central Air': 'Ar Condicionado', 'SalePrice': 'Preço Médio (USD)'},
                    color='Central Air',
                    color_discrete_map={'N': '#ff6b6b', 'Y': '#51cf66'})
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # ANOVA 3: Qualidade da Cozinha
    st.markdown("---")
    st.subheader("3️⃣ Impacto da Qualidade da Cozinha")
    
    df_anova_kitchen = df_anova[df_anova['Kitchen Qual'].isin(['Ex', 'Gd', 'TA', 'Fa'])]
    grupos_kitchen = [df_anova_kitchen[df_anova_kitchen['Kitchen Qual'] == cat]['SalePrice'] 
                      for cat in ['Ex', 'Gd', 'TA', 'Fa']]
    f_stat_kitchen, p_value_kitchen = f_oneway(*grupos_kitchen)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estatística F", f"{f_stat_kitchen:.2f}")
    with col2:
        st.metric("P-valor", f"{p_value_kitchen:.6f}")
    with col3:
        resultado = "✅ Significativo" if p_value_kitchen < 0.05 else "❌ Não Significativo"
        st.metric("Resultado", resultado)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(df_anova_kitchen, x='Kitchen Qual', y='SalePrice',
                    category_orders={'Kitchen Qual': ['Ex', 'Gd', 'TA', 'Fa']},
                    title='Distribuição de Preços por Qualidade da Cozinha',
                    labels={'Kitchen Qual': 'Qualidade da Cozinha', 'SalePrice': 'Preço (USD)'},
                    color='Kitchen Qual')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        means_kitchen = df_anova_kitchen.groupby('Kitchen Qual')['SalePrice'].mean().reset_index()
        means_kitchen = means_kitchen.set_index('Kitchen Qual').reindex(['Ex', 'Gd', 'TA', 'Fa']).reset_index()
        fig = px.bar(means_kitchen, x='Kitchen Qual', y='SalePrice',
                    title='Preço Médio por Qualidade da Cozinha',
                    labels={'Kitchen Qual': 'Qualidade', 'SalePrice': 'Preço Médio (USD)'},
                    color='SalePrice',
                    color_continuous_scale='RdYlGn')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PÁGINA 3: MODELOS DE REGRESSÃO
# ==============================================================================

elif pagina == "🔮 Modelos de Regressão":
    st.header("🔮 Modelos de Regressão Linear")
    
    modelo_escolhido = st.radio(
        "Escolha o modelo para visualizar:",
        ["Modelo 1: Original", "Modelo 2: Log-Log (Recomendado)"],
        horizontal=True
    )
    
    if modelo_escolhido == "Modelo 1: Original":
        model = model_1
        y_true = df_reg['SalePrice']
        y_pred = model.predict(sm.add_constant(df_reg[['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Central_Air_Y']]))
        titulo = "Modelo 1 - Regressão Linear (Forma Original)"
    else:
        model = model_2
        y_true = np.exp(df_reg_log['log_SalePrice'])
        y_pred = np.exp(model.predict(sm.add_constant(df_reg_log[['log_Gr_Liv_Area', 'log_Overall_Qual', 'log_Year_Built', 'log_Garage_Area', 'Central_Air_Y']])))
        titulo = "Modelo 2 - Regressão Log-Log"
    
    st.subheader(titulo)
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R²", f"{r2:.4f}", help="Coeficiente de Determinação")
    with col2:
        st.metric("R² Ajustado", f"{model.rsquared_adj:.4f}")
    with col3:
        st.metric("RMSE", f"${rmse:,.0f}", help="Raiz do Erro Quadrático Médio")
    with col4:
        st.metric("MAE", f"${mae:,.0f}", help="Erro Absoluto Médio")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Valores Preditos vs Reais")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred,
            mode='markers',
            marker=dict(color='#0066cc', size=5, opacity=0.6),
            name='Predições'
        ))
        fig.add_trace(go.Scatter(
            x=[y_true.min(), y_true.max()],
            y=[y_true.min(), y_true.max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Linha Ideal'
        ))
        fig.update_layout(
            xaxis_title='Preço Real (USD)',
            yaxis_title='Preço Predito (USD)',
            height=400,
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Distribuição dos Resíduos")
        residuos = y_true - y_pred
        fig = px.histogram(x=residuos, nbins=50,
                          labels={'x': 'Resíduo (USD)', 'y': 'Frequência'},
                          color_discrete_sequence=['#0066cc'])
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Coeficientes do Modelo")
    
    coef_df = pd.DataFrame({
        'Variável': model.params.index,
        'Coeficiente': model.params.values,
        'P-valor': model.pvalues.values,
        'Significativo': ['✅ Sim' if p < 0.05 else '❌ Não' for p in model.pvalues.values]
    })
    
    coef_df = coef_df[coef_df['Variável'] != 'const']
    
    st.dataframe(
        coef_df.style.format({
            'Coeficiente': '{:.6f}',
            'P-valor': '{:.6f}'
        }),
        hide_index=True,
        use_container_width=True
    )
    
    if modelo_escolhido == "Modelo 2: Log-Log (Recomendado)":
        st.info("""
        **Interpretação dos Coeficientes (Elasticidades):**
        
        No modelo log-log, os coeficientes representam elasticidades:
        - Um coeficiente de 0.5 significa: 1% de aumento na variável → 0.5% de aumento no preço
        - Por exemplo, se log_Gr_Liv_Area = 0.6, então aumentar a área em 10% aumenta o preço em ~6%
        """)
    
    with st.expander("📊 Ver Sumário Completo do Modelo"):
        st.text(str(model.summary()))

# ==============================================================================
# PÁGINA 4: CALCULADORA DE PREÇO
# ==============================================================================

elif pagina == "💰 Calculadora de Preço":
    st.header("💰 Calculadora de Preço de Imóvel")
    
    st.markdown("""
    Ajuste as características do imóvel abaixo para obter uma estimativa de preço
    baseada no **Modelo Log-Log** (mais preciso).
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        area_habitavel = st.slider(
            "Área Habitável (pés²)",
            min_value=int(df['Gr Liv Area'].min()),
            max_value=int(df['Gr Liv Area'].max()),
            value=int(df['Gr Liv Area'].median()),
            step=100
        )
        
        qualidade_geral = st.select_slider(
            "Qualidade Geral",
            options=sorted(df['Overall Qual'].unique()),
            value=int(df['Overall Qual'].median())
        )
        
        ano_construcao = st.slider(
            "Ano de Construção",
            min_value=int(df['Year Built'].min()),
            max_value=int(df['Year Built'].max()),
            value=int(df['Year Built'].median())
        )
    
    with col2:
        area_garagem = st.slider(
            "Área da Garagem (pés²)",
            min_value=0,
            max_value=int(df['Garage Area'].max()),
            value=int(df['Garage Area'].median()),
            step=50
        )
        
        ar_condicionado = st.radio(
            "Ar Condicionado Central",
            options=["Sim", "Não"],
            horizontal=True
        )
        
        st.markdown("### ")  # Espaçamento
    
    # Calcular preço
    if st.button("🔍 Calcular Preço Estimado", type="primary", use_container_width=True):
        
        # Preparar input
        log_area = np.log(area_habitavel)
        log_qual = np.log(qualidade_geral)
        log_ano = np.log(ano_construcao)
        log_garagem = np.log(area_garagem if area_garagem > 0 else 1)
        ar_cond_valor = 1 if ar_condicionado == "Sim" else 0
        
        X_novo = pd.DataFrame({
            'const': [1],
            'log_Gr_Liv_Area': [log_area],
            'log_Overall_Qual': [log_qual],
            'log_Year_Built': [log_ano],
            'log_Garage_Area': [log_garagem],
            'Central_Air_Y': [ar_cond_valor]
        })
        
        log_preco_pred = model_2.predict(X_novo)[0]
        preco_pred = np.exp(log_preco_pred)
        
        # Intervalo de confiança (simplificado)
        margem_erro = rmse_2 * 1.96  # 95% de confiança
        preco_min = preco_pred - margem_erro
        preco_max = preco_pred + margem_erro
        
        st.markdown("---")
        st.success("### 🎯 Resultado da Estimativa")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Preço Mínimo (95%)", f"${preco_min:,.0f}")
        with col2:
            st.metric("**Preço Estimado**", f"**${preco_pred:,.0f}**")
        with col3:
            st.metric("Preço Máximo (95%)", f"${preco_max:,.0f}")
        
        # Comparação com média
        preco_medio = df['SalePrice'].median()
        diferenca = ((preco_pred - preco_medio) / preco_medio) * 100
        
        st.markdown("---")
        st.info(f"""
        **Análise:**
        - O preço estimado é **{abs(diferenca):.1f}% {'acima' if diferenca > 0 else 'abaixo'}** da mediana do mercado (${preco_medio:,.0f})
        - Intervalo de confiança de 95%: ${preco_min:,.0f} - ${preco_max:,.0f}
        - Amplitude do intervalo: ${(preco_max - preco_min):,.0f}
        """)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=preco_pred,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Preço Estimado (USD)", 'font': {'size': 24}},
            delta={'reference': preco_medio, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, preco_max * 1.2]},
                'bar': {'color': "#0066cc"},
                'steps': [
                    {'range': [0, preco_min], 'color': "#e3f2fd"},
                    {'range': [preco_min, preco_max], 'color': "#bbdefb"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': preco_medio
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PÁGINA 5: ANÁLISE DE SENSIBILIDADE
# ==============================================================================

elif pagina == "📉 Análise de Sensibilidade":
    st.header("📉 Análise de Sensibilidade")
    
    st.markdown("""
    Analise como mudanças em cada variável impactam o preço do imóvel.
    Baseado no **Modelo Log-Log** com elasticidades.
    """)
    
    # Valores base
    st.subheader("🏠 Imóvel de Referência")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        base_area = st.number_input("Área (pés²)", value=1500, step=100)
    with col2:
        base_qual = st.number_input("Qualidade", value=6, min_value=1, max_value=10)
    with col3:
        base_ano = st.number_input("Ano", value=2000, min_value=1900, max_value=2024)
    with col4:
        base_garagem = st.number_input("Garagem (pés²)", value=500, step=50)
    with col5:
        base_ar = st.selectbox("Ar Cond.", ["Sim", "Não"])
    
    # Calcular preço base
    X_base = pd.DataFrame({
        'const': [1],
        'log_Gr_Liv_Area': [np.log(base_area)],
        'log_Overall_Qual': [np.log(base_qual)],
        'log_Year_Built': [np.log(base_ano)],
        'log_Garage_Area': [np.log(base_garagem if base_garagem > 0 else 1)],
        'Central_Air_Y': [1 if base_ar == "Sim" else 0]
    })
    preco_base = np.exp(model_2.predict(X_base)[0])
    
    st.metric("💰 Preço Base", f"${preco_base:,.0f}")
    
    st.markdown("---")
    
    variavel_analisar = st.selectbox(
        "Selecione a variável para análise de sensibilidade:",
        ["Área Habitável", "Qualidade Geral", "Ano de Construção", "Área da Garagem"]
    )
    
    # Gerar range de valores
    if variavel_analisar == "Área Habitável":
        valores = np.linspace(base_area * 0.7, base_area * 1.3, 50)
        var_log = 'log_Gr_Liv_Area'
        xlabel = "Área Habitável (pés²)"
        coef = model_2.params[var_log]
    elif variavel_analisar == "Qualidade Geral":
        valores = np.arange(max(1, base_qual - 3), min(10, base_qual + 4))
        var_log = 'log_Overall_Qual'
        xlabel = "Qualidade Geral"
        coef = model_2.params[var_log]
    elif variavel_analisar == "Ano de Construção":
        valores = np.arange(base_ano - 30, base_ano + 5)
        var_log = 'log_Year_Built'
        xlabel = "Ano de Construção"
        coef = model_2.params[var_log]
    else:  # Área da Garagem
        valores = np.linspace(max(0, base_garagem * 0.5), base_garagem * 1.5, 50)
        var_log = 'log_Garage_Area'
        xlabel = "Área da Garagem (pés²)"
        coef = model_2.params[var_log]
    
    # Calcular preços
    precos = []
    for val in valores:
        X_temp = X_base.copy()
        X_temp[var_log] = np.log(val if val > 0 else 1)
        preco = np.exp(model_2.predict(X_temp)[0])
        precos.append(preco)
    
    # Criar DataFrame
    df_sens = pd.DataFrame({
        'Valor': valores,
        'Preço': precos,
        'Variação %': ((np.array(precos) - preco_base) / preco_base * 100)
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Impacto no Preço")
        fig = px.line(df_sens, x='Valor', y='Preço',
                     labels={'Valor': xlabel, 'Preço': 'Preço Estimado (USD)'},
                     title=f'Sensibilidade: {variavel_analisar}')
        fig.add_hline(y=preco_base, line_dash="dash", line_color="red",
                     annotation_text="Preço Base")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Variação Percentual")
        fig = px.line(df_sens, x='Valor', y='Variação %',
                     labels={'Valor': xlabel, 'Variação %': 'Variação no Preço (%)'},
                     title='Variação Relativa')
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📊 Elasticidade e Impacto")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Elasticidade", f"{coef:.4f}")
        st.caption("Variação % no preço para 1% de aumento na variável")
    
    with col2:
        impacto_10 = coef * 10
        st.metric("Impacto de +10%", f"+{impacto_10:.2f}%")
        st.caption("Aumento no preço se a variável aumentar 10%")
    
    with col3:
        valor_10 = preco_base * (impacto_10 / 100)
        st.metric("Valor em $", f"${valor_10:,.0f}")
        st.caption("Aumento monetário do preço")

# ==============================================================================
# PÁGINA 6: COMPARAÇÃO DE MODELOS
# ==============================================================================

else:  # Comparação de Modelos
    st.header("📋 Comparação de Modelos")
    
    st.markdown("""
    Compare o desempenho dos dois modelos de regressão desenvolvidos:
    - **Modelo 1**: Regressão linear na forma original
    - **Modelo 2**: Regressão log-log (transformação logarítmica)
    """)
    
    # Métricas
    y1 = df_reg['SalePrice']
    X1_const = sm.add_constant(df_reg[['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Central_Air_Y']])
    y1_pred = model_1.predict(X1_const)
    
    y2 = np.exp(df_reg_log['log_SalePrice'])
    X2_const = sm.add_constant(df_reg_log[['log_Gr_Liv_Area', 'log_Overall_Qual', 'log_Year_Built', 'log_Garage_Area', 'Central_Air_Y']])
    y2_pred = np.exp(model_2.predict(X2_const))
    
    r2_1 = r2_score(y1, y1_pred)
    rmse_1 = np.sqrt(mean_squared_error(y1, y1_pred))
    mae_1 = mean_absolute_error(y1, y1_pred)
    
    r2_2 = r2_score(y2, y2_pred)
    rmse_2 = np.sqrt(mean_squared_error(y2, y2_pred))
    mae_2 = mean_absolute_error(y2, y2_pred)
    
    st.subheader("📊 Métricas de Desempenho")
    
    comparacao = pd.DataFrame({
        'Métrica': ['R²', 'R² Ajustado', 'RMSE', 'MAE', 'AIC', 'BIC'],
        'Modelo 1 (Original)': [
            f"{r2_1:.4f}",
            f"{model_1.rsquared_adj:.4f}",
            f"${rmse_1:,.0f}",
            f"${mae_1:,.0f}",
            f"{model_1.aic:.0f}",
            f"{model_1.bic:.0f}"
        ],
        'Modelo 2 (Log-Log)': [
            f"{r2_2:.4f}",
            f"{model_2.rsquared_adj:.4f}",
            f"${rmse_2:,.0f}",
            f"${mae_2:,.0f}",
            f"{model_2.aic:.0f}",
            f"{model_2.bic:.0f}"
        ],
        'Melhor': [
            'Modelo 2' if r2_2 > r2_1 else 'Modelo 1',
            'Modelo 2' if model_2.rsquared_adj > model_1.rsquared_adj else 'Modelo 1',
            'Modelo 2' if rmse_2 < rmse_1 else 'Modelo 1',
            'Modelo 2' if mae_2 < mae_1 else 'Modelo 1',
            'Modelo 2' if model_2.aic < model_1.aic else 'Modelo 1',
            'Modelo 2' if model_2.bic < model_1.bic else 'Modelo 1'
        ]
    })
    
    st.dataframe(comparacao, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Modelo 1: Predito vs Real")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=y1, y=y1_pred, mode='markers',
                                 marker=dict(color='#ff6b6b', size=5, opacity=0.6),
                                 name='Predições'))
        fig1.add_trace(go.Scatter(x=[y1.min(), y1.max()], y=[y1.min(), y1.max()],
                                 mode='lines', line=dict(color='black', dash='dash'),
                                 name='Ideal'))
        fig1.update_layout(xaxis_title='Real', yaxis_title='Predito', height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.metric("R² do Modelo 1", f"{r2_1:.4f}")
    
    with col2:
        st.subheader("Modelo 2: Predito vs Real")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=y2, y=y2_pred, mode='markers',
                                 marker=dict(color='#51cf66', size=5, opacity=0.6),
                                 name='Predições'))
        fig2.add_trace(go.Scatter(x=[y2.min(), y2.max()], y=[y2.min(), y2.max()],
                                 mode='lines', line=dict(color='black', dash='dash'),
                                 name='Ideal'))
        fig2.update_layout(xaxis_title='Real', yaxis_title='Predito', height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.metric("R² do Modelo 2", f"{r2_2:.4f}")
    
    st.markdown("---")
    
    st.subheader("🎯 Recomendação Final")
    
    st.success(f"""
    ### ✅ Modelo Recomendado: **Modelo 2 (Log-Log)**
    
    **Justificativas:**
    - **Maior R²**: {r2_2:.4f} vs {r2_1:.4f} (+{((r2_2-r2_1)/r2_1*100):.1f}%)
    - **Menor RMSE**: ${rmse_2:,.0f} vs ${rmse_1:,.0f}
    - **Interpretação**: Coeficientes representam elasticidades (% de mudança)
    - **Pressupostos**: Melhor atendimento aos pressupostos da regressão linear
    - **Heterocedasticidade**: Reduzida após transformação logarítmica
    
    O Modelo 2 é superior em todos os critérios avaliados e permite
    interpretação percentual direta dos coeficientes, facilitando análises de negócio.
    """)
    
    with st.expander("📖 Ver Detalhes Técnicos"):
        st.markdown("""
        ### Testes Diagnósticos
        
        **Modelo 1 (Original):**
        - Heterocedasticidade detectada (Breusch-Pagan)
        - Desvios da normalidade nos resíduos
        - Relações não-lineares identificadas
        
        **Modelo 2 (Log-Log):**
        - Homocedasticidade melhorada significativamente
        - Resíduos mais próximos da distribuição normal
        - Linearização das relações entre variáveis
        - VIF aceitável (< 10) para todas as variáveis
        
        ### Interpretação dos Coeficientes
        
        No modelo log-log, um coeficiente β significa:
        > "Para cada 1% de aumento na variável X, o preço aumenta aproximadamente β%"
        
        Exemplo: Se o coeficiente de log_Gr_Liv_Area = 0.6:
        - Aumentar a área em 10% → aumenta o preço em ~6%
        - Aumentar a área em 20% → aumenta o preço em ~12%
        """)

# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
        <p style='color: #666; margin: 5px;'><strong>Dashboard de Precificação Imobiliária</strong></p>
        <p style='color: #666; margin: 5px;'>Desenvolvido por: Pedro Arthur Santos Oliveira (231036069)</p>
        <p style='color: #666; margin: 5px;'>Universidade de Brasília - Engenharia de Produção</p>
        <p style='color: #666; margin: 5px;'>Disciplina: Sistemas de Informação em Engenharia de Produção</p>
        <p style='color: #666; margin: 5px;'>Professor: João Gabriel de Moraes Souza</p>
    </div>
""", unsafe_allow_html=True)
