# 🚔 Delegacia 5.0 — Plataforma de Análise Preditiva de Crimes

### Projeto Integrador 2025.2 — Faculdade Senac Pernambuco  

---

## 📘 Descrição do Projeto

O **Delegacia Dashboard** é uma aplicação de análise criminal interativa desenvolvida em **Streamlit**, combinando **modelos supervisionados e não supervisionados de IA** para apoiar equipes de investigação da **Polícia Civil de Pernambuco (PC-PE)**.  
A plataforma oferece **insights preditivos, detecção de padrões e anomalias**, além de visualizações intuitivas para **tomada de decisão em segurança pública**.

---

## 🎯 Objetivo do Projeto

Construir uma **prova de conceito (PoC)** de uma solução analítica capaz de:
- Classificar ocorrências criminais por tipo;
- Identificar **padrões espaciais e temporais** (hotspots);
- Detectar **anomalias e agrupamentos** de crimes similares;
- Oferecer um **dashboard visual** para exploração interativa dos resultados.

---

## 🧩 Funcionalidades Principais

- 📊 **Dashboard interativo (Streamlit)**
  - Filtros dinâmicos por local, tipo e período;
  - Mapas de calor de regiões críticas;
  - Exibição de métricas e gráficos de tendência.

- 🧠 **IA Supervisionada**
  - Modelos de classificação (Random Forest, XGBoost);
  - Avaliação com métricas de precisão, recall e F1.

- 🧭 **IA Não Supervisionada**
  - Clusterização (KMeans, DBSCAN, HDBSCAN);
  - Detecção de anomalias com Isolation Forest.

- 🔍 **Storytelling de Dados**
  - Relatórios automáticos com insights e limitações do modelo.

- 🔒 **Conformidade com LGPD**
  - Dados anonimizados e uso ético das informações.

---

## ⚙️ Tecnologias Utilizadas

| Categoria | Ferramentas |
|------------|-------------|
| Linguagem | Python 3.11 |
| Dashboard | Streamlit |
| Modelagem | Scikit-learn, XGBoost, LightGBM |
| Visualização | Plotly, Matplotlib, PyDeck |
| Clusterização | KMeans, DBSCAN, HDBSCAN |
| Explicabilidade | SHAP |
| Outras | Pandas, NumPy, Geopandas |

---

## 🚀 Como Executar Localmente

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/IchinoSanji/delegacia-dashboard.git
cd delegacia-dashboard
```

### 2️⃣ Criar ambiente virtual
```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
```

### 3️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

### 4️⃣ Executar o dashboard
```bash
streamlit run app.py
```

Acesse o app no navegador:  
👉 **http://localhost:8501**

---

## ☁️ Deploy da Aplicação

Você pode publicar o dashboard em diferentes plataformas gratuitas de forma simples.

### 🟦 **Opção 1 — Streamlit Cloud (recomendada)**
1. Acesse [https://share.streamlit.io](https://share.streamlit.io)
2. Conecte sua conta do GitHub.
3. Escolha o repositório **IchinoSanji/delegacia-dashboard**.
4. Configure:
   - **Main file path:** `app.py`
   - **Python version:** `3.11`
   - **Requirements file:** `requirements.txt`
5. Clique em **Deploy** e aguarde alguns minutos.
6. Seu app estará disponível em:  
   `https://seuusuario.streamlit.app`

### 🟪 **Opção 2 — Render**
1. Vá até [https://render.com](https://render.com)
2. Crie um novo **Web Service**.
3. Selecione o repositório GitHub.
4. Configure:
   - Start command:  
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```
   - Environment: Python 3.11
5. Salve e publique o serviço.

### 🟨 **Opção 3 — Hugging Face Spaces**
1. Vá para [https://huggingface.co/spaces](https://huggingface.co/spaces)
2. Crie um novo *Space* → Tipo: **Streamlit**
3. Faça o upload do código do projeto.
4. Aguarde o build automático.
5. O app será disponibilizado em segundos.

---

## 🧮 Modelos e Métricas

| Tipo | Algoritmo | Objetivo | Métricas |
|------|------------|-----------|-----------|
| Supervisionado | Random Forest, XGBoost | Classificação de crimes | Precision, Recall, F1, ROC-AUC |
| Não Supervisionado | KMeans, DBSCAN | Agrupamento de ocorrências | Silhouette, DBCV |
| Anomalias | Isolation Forest | Detecção de casos fora do padrão | True Positive Rate |
| Explicabilidade | SHAP | Interpretação do modelo | Feature Importance |

---

## 📈 Resultados e Insights

- Identificação de **zonas de maior risco** por hora e bairro;  
- Descoberta de **padrões temporais** (ex.: aumento noturno de furtos);  
- Visualização de **agrupamentos criminais** por modus operandi;  
- Explicabilidade dos modelos com **importância de variáveis**.

---

## 💡 Possíveis Expansões

- Simulação de **alocação de patrulhas** com aprendizado por reforço;  
- Integração com APIs geográficas e socioeconômicas;  
- Exportação automática de relatórios em PDF/HTML;  
- Deploy híbrido (Web + Mobile).

---

## 👥 Equipe do Projeto

- Abraão Saraiva 
- Carlos Henrique
- Klara Marinho
- Lucas Eduardo
- Luiz Reis

---

## 📄 Licença

Projeto desenvolvido exclusivamente para fins **acadêmicos** no contexto do **Projeto Integrador 5 — Faculdade Senac PE**, respeitando as diretrizes de **ética, LGPD e segurança da informação**.

---

## 📚 Referências

- Desafio oficial: [Delegacia 5.0 — Solução tecnológica para análise preditiva de crimes](https://desafios.pe.gov.br/challenge?url=projeto-delegacia-50-solucao-tecnologica-para-analise-preditiva-de-crimes)

---

### 🧭 Desenvolvido com propósito social e foco em inovação para a segurança pública de Pernambuco.
