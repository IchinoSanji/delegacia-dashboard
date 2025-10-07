# -*- coding: utf-8 -*-
"""
Delegacia 5.0 — Painel de Risco Operacional

Foco: identificar e prever ocorrências de **alto risco para o policial** (arma de fogo / explosivos),
dando ênfase a **onde** (bairro), **quando** (dia/horário) e **por quê** (fatores que mais pesam),
além de mostrar o **ganho operacional** via F1@20%.

Como executar localmente:
    1) pip install -r requirements.txt
    2) streamlit run app_risco_operacional.py

Pré‑requisitos no CSV (dataset_ocorrencias_delegacia_5.csv):
    - 'data_ocorrencia' (data/hora)
    - 'bairro'
    - 'tipo_crime'
    - 'arma_utilizada' (com valores como 'Arma de Fogo' e 'Explosivos')
    - Opcional: 'latitude', 'longitude', 'quantidade_vitimas', 'quantidade_suspeitos', 'idade_suspeito', 'sexo_suspeito'
"""

from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, roc_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Mapa (opcional)
try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

RANDOM_STATE = 42
CSV_PATH = "dataset_ocorrencias_delegacia_5.csv"

st.set_page_config(page_title="Delegacia 5.0 — Risco Operacional", layout="wide")
st.title("🚓 Delegacia 5.0 — Painel de Risco Operacional")
st.caption("Foco no risco ao policial: onde, quando e com quais fatores priorizar apoio tático.")

# -------------------------
# Carregamento e preparação de dados
# -------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Parse de data/hora
    if 'data_ocorrencia' in df.columns:
        df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'], errors='coerce')
    else:
        # fallback: tentar detectar alguma coluna de data
        for c in df.columns:
            if 'data' in c.lower():
                df['data_ocorrencia'] = pd.to_datetime(df[c], errors='coerce')
                break
    # Derivados temporais
    if 'data_ocorrencia' in df.columns:
        df['ano'] = df['data_ocorrencia'].dt.year
        df['mes'] = df['data_ocorrencia'].dt.to_period('M').astype(str)
        df['hora'] = df['data_ocorrencia'].dt.hour
        df['dia_semana_idx'] = df['data_ocorrencia'].dt.dayofweek # 0=seg ... 6=dom
        df['dia_semana'] = df['data_ocorrencia'].dt.day_name()
        df['turno'] = df['hora'].apply(lambda h: (
            'Madrugada' if h is not np.nan and 0 <= int(h) < 6 else
            'Manhã' if h is not np.nan and 6 <= int(h) < 12 else
            'Tarde' if h is not np.nan and 12 <= int(h) < 18 else
            'Noite' if h is not np.nan and 18 <= int(h) <= 23 else
            'Desconhecido'))
    # Alvo: risco alto se arma de fogo ou explosivos
    if 'arma_utilizada' in df.columns:
        df['risco_alto'] = df['arma_utilizada'].isin(['Arma de Fogo', 'Explosivos']).astype(int)
    else:
        df['risco_alto'] = 0

    # Limpeza básica de categóricos
    for c in ['bairro', 'tipo_crime', 'sexo_suspeito', 'descricao_modus_operandi']:
        if c in df.columns:
            df[c] = df[c].fillna('Ignorado')

    # Garantir numéricos onde aplicável
    for c in ['latitude', 'longitude', 'quantidade_vitimas', 'quantidade_suspeitos', 'idade_suspeito', 'hora', 'dia_semana_idx']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df

try:
    df = load_data(CSV_PATH)
    st.success(f"Dataset carregado ({len(df):,} registros).")
except Exception as e:
    st.error("Não foi possível carregar o dataset. Verifique o caminho e o formato.")
    st.exception(e)
    st.stop()

# -------------------------
# Filtros (sidebar)
# -------------------------
st.sidebar.header("🎛️ Filtros")
if 'data_ocorrencia' in df.columns and df['data_ocorrencia'].notna().any():
    min_date, max_date = df['data_ocorrencia'].min(), df['data_ocorrencia'].max()
    date_range = st.sidebar.date_input("Período", [min_date.date(), max_date.date()])
else:
    date_range = None

bairros = ["Todos"] + (sorted(df['bairro'].dropna().unique().tolist()) if 'bairro' in df.columns else [])
tipos = ["Todos"] + (sorted(df['tipo_crime'].dropna().unique().tolist()) if 'tipo_crime' in df.columns else [])
turnos_opts = ["Madrugada","Manhã","Tarde","Noite"]

bairro_sel = st.sidebar.selectbox("Bairro", bairros, index=0)
tipo_sel = st.sidebar.selectbox("Tipo de crime", tipos, index=0)
turnos_sel = st.sidebar.multiselect("Turno (opcional)", options=turnos_opts, default=[])

# Aplicar filtros
view = df.copy()
if date_range and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    view = view[(view['data_ocorrencia'] >= start) & (view['data_ocorrencia'] <= end + pd.Timedelta(days=1))]
if bairro_sel != "Todos" and 'bairro' in view.columns:
    view = view[view['bairro'] == bairro_sel]
if tipo_sel != "Todos" and 'tipo_crime' in view.columns:
    view = view[view['tipo_crime'] == tipo_sel]
if turnos_sel and 'turno' in view.columns:
    view = view[view['turno'].isin(turnos_sel)]

# -------------------------
# Funções auxiliares (modelo, F1@k etc.)
# -------------------------

def temporal_splits(d: pd.DataFrame, target: str = 'risco_alto'):
    """Split temporal (60/25/15% por quantis de data). Fallback: split aleatório."""
    if 'data_ocorrencia' in d.columns and d['data_ocorrencia'].notna().any():
        q60 = d['data_ocorrencia'].quantile(0.60)
        q85 = d['data_ocorrencia'].quantile(0.85)
        train = d[d['data_ocorrencia'] <= q60]
        valid = d[(d['data_ocorrencia'] > q60) & (d['data_ocorrencia'] <= q85)]
        test  = d[d['data_ocorrencia'] > q85]
        # Fallbacks
        if len(train) < 50 or len(valid) < 20:
            train, test = train_test_split(d, test_size=0.25, random_state=RANDOM_STATE,
                                           stratify=d[target] if d[target].nunique() > 1 else None)
            valid = test.copy()
    else:
        train, valid = train_test_split(d, test_size=0.3, random_state=RANDOM_STATE,
                                        stratify=d[target] if d[target].nunique() > 1 else None)
        test = valid.copy()
    return train, valid, test


def build_and_choose_model(train: pd.DataFrame, valid: pd.DataFrame, target: str = 'risco_alto'):
    num_cols, cat_cols = [], []
    for c in ['quantidade_vitimas','quantidade_suspeitos','idade_suspeito','hora','dia_semana_idx']:
        if c in train.columns:
            num_cols.append(c)
    for c in ['bairro','tipo_crime','sexo_suspeito','turno']:
        if c in train.columns:
            cat_cols.append(c)

    X_train, y_train = train[num_cols + cat_cols].copy(), train[target].copy()
    X_valid, y_valid = valid[num_cols + cat_cols].copy(), valid[target].copy() if target in valid else None

    num_pipe = Pipeline([('scaler', StandardScaler())]) if len(num_cols) > 0 else 'drop'
    cat_pipe = Pipeline([('oh', OneHotEncoder(handle_unknown='ignore'))]) if len(cat_cols) > 0 else 'drop'

    prep = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ], remainder='drop')

    pipe_lr = Pipeline([('prep', prep), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE))])
    pipe_rf = Pipeline([('prep', prep), ('clf', RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE))])

    pipe_lr.fit(X_train, y_train)
    pipe_rf.fit(X_train, y_train)

    def best_thr(y, p):
        grid = np.linspace(0.2, 0.8, 61)
        f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
        i = int(np.argmax(f1s))
        return float(grid[i]), float(f1s[i])

    chosen, thr = pipe_rf, 0.5
    chosen_name = "RandomForest"
    if y_valid is not None and len(y_valid) > 0:
        p_lr = pipe_lr.predict_proba(X_valid)[:,1]
        p_rf = pipe_rf.predict_proba(X_valid)[:,1]
        thr_lr, f1_lr = best_thr(y_valid, p_lr)
        thr_rf, f1_rf = best_thr(y_valid, p_rf)
        if f1_rf >= f1_lr:
            chosen, thr, chosen_name = pipe_rf, thr_rf, "RandomForest"
        else:
            chosen, thr, chosen_name = pipe_lr, thr_lr, "LogisticRegression"

    return chosen, thr, chosen_name, (num_cols, cat_cols)


def evaluate_model(model: Pipeline, thr: float, test: pd.DataFrame, target: str = 'risco_alto', num_cat_cols=None):
    num_cols, cat_cols = (num_cat_cols or ([], []))
    X_test = test[num_cols + cat_cols]
    y_test = test[target]
    prob = model.predict_proba(X_test)[:,1]
    y_hat = (prob >= thr).astype(int)
    metrics = {
        'precision': precision_score(y_test, y_hat, zero_division=0),
        'recall': recall_score(y_test, y_hat, zero_division=0),
        'f1': f1_score(y_test, y_hat, zero_division=0),
        'roc_auc': roc_auc_score(y_test, prob) if y_test.nunique() > 1 else np.nan
    }
    return metrics, prob


def f1_at_k_by_window(df_part: pd.DataFrame, probs: np.ndarray, k_pct: float = 0.20, window_cols=("bairro","mes")) -> pd.DataFrame:
    d = df_part.copy()
    d = d.assign(prob=probs, y=d['risco_alto'].astype(int))
    frames = []
    for key, g in d.groupby(list(window_cols)):
        if len(g) == 0:
            continue
        k = max(1, int(len(g) * k_pct))
        g_sorted = g.sort_values('prob', ascending=False)
        picked = g_sorted.head(k).copy(); picked['y_hat'] = 1
        rest = g_sorted.iloc[k:].copy(); rest['y_hat'] = 0
        gg = pd.concat([picked, rest], ignore_index=True)
        p = precision_score(gg['y'], gg['y_hat'], zero_division=0)
        r = recall_score(gg['y'], gg['y_hat'], zero_division=0)
        f = f1_score(gg['y'], gg['y_hat'], zero_division=0)
        frames.append(pd.DataFrame([{'window': key, 'n': len(g), 'precision': p, 'recall': r, 'f1': f}]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['window','n','precision','recall','f1'])

# -------------------------
# Tabs
# -------------------------
TAB1, TAB2, TAB3, TAB4 = st.tabs(["Visão Operacional", "Predição de Risco", "Desempenho do Modelo", "Relatório Inteligente"])

# =========================
# 1) Visão Operacional
# =========================
with TAB1:
    st.header("Panorama de Risco ao Policial")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Ocorrências (filtro)", f"{len(view):,}")
    with c2:
        st.metric("Taxa média de alto risco", f"{(view['risco_alto'].mean() * 100 if len(view)>0 else 0):.1f}%")
    with c3:
        st.metric("Bairros distintos", view['bairro'].nunique() if 'bairro' in view.columns else "—")
    with c4:
        if 'data_ocorrencia' in view.columns and view['data_ocorrencia'].notna().any():
            st.metric("Período analisado", f"{view['data_ocorrencia'].min().date()} → {view['data_ocorrencia'].max().date()}")
        else:
            st.metric("Período analisado", "—")

    # Top bairros por risco
    st.subheader("Top 10 bairros por risco médio")
    if 'bairro' in view.columns and len(view) > 0:
        top_bairros = (view.groupby('bairro')['risco_alto']
                        .mean().sort_values(ascending=False).head(10).reset_index())
        top_bairros['Risco (%)'] = (top_bairros['risco_alto'] * 100).round(1)
        st.dataframe(top_bairros[['bairro','Risco (%)']], use_container_width=True)
    else:
        st.info("Sem dados de 'bairro' após filtros.")

    # Heatmap: dia_semana x hora (proporção de risco)
    st.subheader("Quando o risco é maior? (Dia da semana × Hora)")
    if {'dia_semana','hora','risco_alto'}.issubset(view.columns) and len(view) > 0:
        # Ordenar os dias
        ordem_dias = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        pivot = (view.assign(dia_cat=pd.Categorical(view['dia_semana'], categories=ordem_dias, ordered=True))
                      .groupby(['dia_cat','hora'])['risco_alto'].mean().reset_index())
        heat = pivot.pivot(index='dia_cat', columns='hora', values='risco_alto').fillna(0)
        fig, ax = plt.subplots()
        im = ax.imshow(heat.values, aspect='auto')
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(['Seg','Ter','Qua','Qui','Sex','Sáb','Dom'])
        ax.set_xticks(range(0,24,2))
        ax.set_xticklabels(list(range(0,24,2)))
        ax.set_xlabel('Hora do dia')
        ax.set_ylabel('Dia da semana')
        ax.set_title('Proporção de alto risco')
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Não há colunas suficientes para o heatmap (dia/horário).")

    # Tipos de crime mais associados ao alto risco
    st.subheader("Quais tipos de crime mais envolvem arma/explosivos?")
    if {'tipo_crime','risco_alto'}.issubset(view.columns) and len(view) > 0:
        tipos_rank = (view.groupby('tipo_crime')['risco_alto']
                        .mean().sort_values(ascending=False).head(12).reset_index())
        tipos_rank['Risco (%)'] = (tipos_rank['risco_alto'] * 100).round(1)
        st.dataframe(tipos_rank[['tipo_crime','Risco (%)']], use_container_width=True)
    else:
        st.info("Colunas 'tipo_crime' e/ou 'risco_alto' indisponíveis após filtros.")

    # Mapa (amostra) — cor por risco
    st.subheader("Mapa (amostra) — concentração de ocorrências e risco")
    if HAS_FOLIUM and {'latitude','longitude'}.issubset(view.columns):
        sample = view.dropna(subset=['latitude','longitude']).sample(min(1200, len(view)), random_state=RANDOM_STATE) if len(view)>0 else view
        if len(sample) > 0:
            m = folium.Map(location=[sample['latitude'].mean(), sample['longitude'].mean()], zoom_start=12)
            for _, r in sample.iterrows():
                color = 'red' if int(r.get('risco_alto',0)) == 1 else 'blue'
                folium.CircleMarker(
                    location=[float(r['latitude']), float(r['longitude'])],
                    radius=3, color=color, fill=True, fill_opacity=0.6,
                    popup=f"{r.get('bairro','')} | {r.get('tipo_crime','')} | Risco:{int(r.get('risco_alto',0))}"
                ).add_to(m)
            st_folium(m, width=900, height=480)
        else:
            st.info("Sem coordenadas disponíveis após filtros.")
    else:
        st.info("Mapa desativado (Folium não disponível ou lat/lon ausentes).")

    # Resumo textual automático
    st.subheader("Resumo automático — linguagem operacional")
    def resumo_operacional(d: pd.DataFrame) -> str:
        if len(d) == 0:
            return "Sem registros no filtro atual."
        linhas = []
        linhas.append(f"Período analisado: {d['data_ocorrencia'].min().date() if 'data_ocorrencia' in d else '—'} → {d['data_ocorrencia'].max().date() if 'data_ocorrencia' in d else '—'}")
        linhas.append(f"Proporção de alto risco: {(d['risco_alto'].mean()*100):.1f}%")
        if 'bairro' in d.columns:
            topb = (d.groupby('bairro')['risco_alto'].mean().sort_values(ascending=False).head(3))
            if len(topb) > 0:
                linhas.append("Bairros mais críticos: " + ", ".join([f"{b} ({v*100:.1f}%)" for b,v in topb.items()]))
        if {'hora','risco_alto'}.issubset(d.columns):
            horas = (d.groupby('hora')['risco_alto'].mean().sort_values(ascending=False))
            if len(horas) > 0:
                htop = list(horas.head(3).index)
                linhas.append("Horários mais arriscados (top 3): " + ", ".join([f"{int(h)}h" for h in htop]))
        if 'tipo_crime' in d.columns:
            tc = (d.groupby('tipo_crime')['risco_alto'].mean().sort_values(ascending=False).head(3))
            if len(tc) > 0:
                linhas.append("Tipos com maior risco: " + ", ".join([f"{k} ({v*100:.1f}%)" for k,v in tc.items()]))
        return "\n".join(linhas)

    st.text(resumo_operacional(view))

# =========================
# 2) Predição de Risco
# =========================
with TAB2:
    st.header("Predição de risco para um novo chamado")

    if 'risco_alto' not in df.columns:
        st.info("Coluna alvo 'risco_alto' ausente — impossível treinar modelo.")
        st.stop()

    train, valid, test = temporal_splits(df, 'risco_alto')
    model, thr, model_name, (num_cols, cat_cols) = build_and_choose_model(train, valid, 'risco_alto')

    st.markdown(f"Modelo escolhido: **{model_name}** | limiar de decisão: **{thr:.2f}**")

    with st.form("novo_caso"):
        inputs = {}
        # Categóricos
        for c in [x for x in ['bairro','tipo_crime','sexo_suspeito','turno'] if x in df.columns]:
            opts = ["(não informar)"] + sorted(df[c].dropna().unique().tolist())
            inputs[c] = st.selectbox(c, options=opts, index=0)
        # Numéricos
        if 'hora' in df.columns:
            inputs['hora'] = st.slider("Hora (0–23)", min_value=0, max_value=23, value=20)
        for c in ['quantidade_vitimas','quantidade_suspeitos','idade_suspeito']:
            if c in df.columns:
                inputs[c] = st.number_input(c, min_value=0, max_value=100, value=0)
        submitted = st.form_submit_button("Estimar risco")

    if submitted:
        row = {}
        for c in num_cols:
            row[c] = [inputs.get(c, 0)]
        for c in cat_cols:
            v = inputs.get(c, None)
            row[c] = [np.nan if (v is None or v == "(não informar)") else v]
        X_new = pd.DataFrame(row)
        for c in cat_cols:
            X_new[c] = X_new[c].fillna('Ignorado')
        proba = float(model.predict_proba(X_new)[0, 1])
        pred = int(proba >= thr)
        st.metric("Probabilidade de alto risco (arma/explosivos)", f"{proba:.1%}")
        st.write("Decisão com base no limiar:", "**ALTO RISCO** — priorizar apoio tático" if pred==1 else "Baixo risco — rotina padrão")
        st.caption("Use como insumo: a confirmação é sempre da equipe operacional.")

# =========================
# 3) Desempenho do Modelo
# =========================
with TAB3:
    st.header("Desempenho, explicabilidade e ganho operacional")

    if 'risco_alto' not in df.columns:
        st.info("Coluna alvo 'risco_alto' ausente — impossível avaliar modelo.")
        st.stop()

    train, valid, test = temporal_splits(df, 'risco_alto')
    model, thr, model_name, (num_cols, cat_cols) = build_and_choose_model(train, valid, 'risco_alto')
    metrics, prob_test = evaluate_model(model, thr, test, 'risco_alto', (num_cols, cat_cols))

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Precision", f"{metrics['precision']:.3f}")
    with c2: st.metric("Recall", f"{metrics['recall']:.3f}")
    with c3: st.metric("F1", f"{metrics['f1']:.3f}")
    with c4: st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if not np.isnan(metrics['roc_auc']) else "—")

    # Curvas ROC e PR
    try:
        y_true = test['risco_alto']
        fpr, tpr, _ = roc_curve(y_true, prob_test)
        prec, rec, _ = precision_recall_curve(y_true, prob_test)

        fig1, ax1 = plt.subplots(); ax1.plot(fpr, tpr); ax1.plot([0,1],[0,1],'--'); ax1.set_title('ROC (teste)'); ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
        st.pyplot(fig1, clear_figure=True)
        fig2, ax2 = plt.subplots(); ax2.plot(rec, prec); ax2.set_title('Precision–Recall (teste)'); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
        st.pyplot(fig2, clear_figure=True)
    except Exception:
        st.info("Não foi possível desenhar ROC/PR para a partição atual.")

    # Importância de features (se RandomForest)
    try:
        clf = model.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            prep = model.named_steps['prep']
            feat_names = []
            feat_names += num_cols
            if 'cat' in prep.named_transformers_:
                oh = prep.named_transformers_['cat'].named_steps.get('oh', None)
                if oh is not None:
                    feat_names += oh.get_feature_names_out(cat_cols).tolist()
            importances = clf.feature_importances_
            order = np.argsort(importances)[::-1][:12]
            imp_df = pd.DataFrame({"feature": [feat_names[i] for i in order], "importance": importances[order]})
            st.subheader("Principais fatores que influenciam o risco")
            st.table(imp_df)
    except Exception:
        st.info("Importância de features não disponível para o modelo atual.")

    # F1@20% por bairro/mês — ganho operacional
    st.subheader("F1@20% por bairro/mês — ganho com priorização do top 20%")
    if {'bairro','mes'}.issubset(test.columns):
        test_with_prob = test.copy(); test_with_prob['prob_risco'] = prob_test
        f1k = f1_at_k_by_window(test_with_prob, prob_test, k_pct=0.20, window_cols=("bairro","mes"))
        if len(f1k) > 0:
            st.write("Média F1@20% (todas as janelas):", f"{f1k['f1'].mean():.3f}")
            # Agregado por bairro
            f1k_bairro = (f1k.groupby(f1k['window'].apply(lambda x: x[0]))['f1'].mean().sort_values(ascending=False).head(12))
            st.dataframe(f1k_bairro.reset_index().rename(columns={'window':'bairro','f1':'F1@20% (médio)'}), use_container_width=True)
        else:
            st.info("Não foi possível calcular F1@20% nas janelas atuais.")
    else:
        st.info("Colunas 'bairro' e/ou 'mes' ausentes para F1@20%.")

# =========================
# 4) Relatório Inteligente
# =========================
with TAB4:
    st.header("Relatório automático de risco")

    def build_report(d: pd.DataFrame, metrics: dict | None = None) -> str:
        linhas = []
        linhas.append("Relatório — Delegacia 5.0 (Risco Operacional)")
        linhas.append(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if 'data_ocorrencia' in d.columns and d['data_ocorrencia'].notna().any():
            linhas.append(f"Período: {d['data_ocorrencia'].min().date()} → {d['data_ocorrencia'].max().date()}")
        linhas.append(f"Registros analisados: {len(d):,}")
        linhas.append(f"Proporção de alto risco (arma/explosivos): {(d['risco_alto'].mean()*100 if len(d)>0 else 0):.1f}%")
        if 'bairro' in d.columns:
            topb = (d.groupby('bairro')['risco_alto'].mean().sort_values(ascending=False).head(5))
            if len(topb) > 0:
                linhas.append("Top 5 bairros (maior risco médio):")
                for b,v in topb.items():
                    linhas.append(f" - {b}: {v*100:.1f}%")
        if 'tipo_crime' in d.columns:
            topt = (d.groupby('tipo_crime')['risco_alto'].mean().sort_values(ascending=False).head(5))
            if len(topt) > 0:
                linhas.append("Top 5 tipos de crime (maior risco médio):")
                for t,v in topt.items():
                    linhas.append(f" - {t}: {v*100:.1f}%")
        if metrics is not None and len(metrics) > 0:
            linhas.append("Desempenho do modelo (teste):")
            for k in ['precision','recall','f1','roc_auc']:
                if k in metrics and metrics[k] is not None and not (isinstance(metrics[k], float) and np.isnan(metrics[k])):
                    linhas.append(f" - {k.upper()}: {metrics[k]:.3f}")
        linhas.append("Observações:")
        linhas.append(" - Use como insumo para priorização; decisão final é da equipe.")
        linhas.append(" - Limitações: possíveis vieses de registro; revisar anomalias e outliers.")
        return "\n".join(linhas)

    # Recalcular desempenho global (sem filtro) para incluir no relatório
    if 'risco_alto' in df.columns:
        tr, va, te = temporal_splits(df, 'risco_alto')
        mdl, thr_, mdl_name, (numc, catc) = build_and_choose_model(tr, va, 'risco_alto')
        met, prob_t = evaluate_model(mdl, thr_, te, 'risco_alto', (numc, catc))
    else:
        met = {}

    report_text = build_report(view, met)
    st.text_area("Relatório (texto)", value=report_text, height=320)
    st.download_button("Baixar relatório (.txt)", data=report_text, file_name="relatorio_risco_operacional.txt", mime="text/plain")
