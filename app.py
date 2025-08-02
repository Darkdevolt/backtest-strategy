import streamlit as st
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import io

# --- Configuration de la page ---
st.set_page_config(page_title="Backtesting BRVM", layout="wide")
st.title("📈 Backtesting de Stratégies sur Actions BRVM")

st.write("Téléversez vos données et ajustez les paramètres pour simuler les performances d'une stratégie d'investissement.")

# --- Panneau latéral ---
with st.sidebar:
    st.header("1. Vos Données")
    uploaded_file = st.file_uploader("Fichier CSV (avec colonnes : Date, Open, High, Low, Close)", type="csv")

    st.header("2. Portefeuille")
    initial_cash = st.number_input("Capital de départ", min_value=100, value=10000, step=100)
    currency_symbol = st.text_input("Symbole de la devise", value="FCFA")

    st.header("3. Choix de la Stratégie")
    strategy_name = st.selectbox("Stratégie à tester", ["Bandes de Bollinger", "Moyenne Mobile", "RSI"])

    if strategy_name == "Bandes de Bollinger":
        st.markdown("**Paramètres Bollinger**")
        bollinger_period = st.slider("Période", 5, 100, 20)
        bollinger_std = st.slider("Écart-type", 1.0, 4.0, 2.0, 0.1)
        optimize = st.checkbox("🔁 Optimiser automatiquement les paramètres")

# --- Chargement des données ---
@st.cache_data
def load_uploaded_data(file):
    if file is None:
        return None
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df.columns = [col.capitalize() for col in df.columns]
        required = {'Date', 'Open', 'High', 'Low', 'Close'}
        if not required.issubset(df.columns):
            st.error("Colonnes requises manquantes : " + ', '.join(required))
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.set_index('Date', inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        return None

data = load_uploaded_data(uploaded_file)

# --- Définition des stratégies ---
class BollingerBandsStrategy(Strategy):
    n_period = 20
    n_std = 2.0
    def init(self):
        price = self.data.Close
        self.sma, self.upper, self.lower = self.I(
            lambda s, n, std: (
                s.rolling(n).mean(),
                s.rolling(n).mean() + std * s.rolling(n).std(),
                s.rolling(n).mean() - std * s.rolling(n).std()
            ), pd.Series(price), self.n_period, self.n_std)
    def next(self):
        if crossover(self.data.Close, self.lower):
            self.buy()
        elif crossover(self.data.Close, self.upper):
            if self.position: self.position.close()

class MovingAverageCrossStrategy(Strategy):
    def init(self):
        self.sma1 = self.I(lambda x: x.rolling(10).mean(), self.data.Close)
        self.sma2 = self.I(lambda x: x.rolling(30).mean(), self.data.Close)
    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            if self.position: self.position.close()

class RSIStrategy(Strategy):
    def init(self):
        close = self.data.Close
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        self.rsi = self.I(lambda rs: 100 - (100 / (1 + rs)), rs)
    def next(self):
        if self.rsi[-1] < 30:
            self.buy()
        elif self.rsi[-1] > 70:
            if self.position: self.position.close()

# --- Exécution du backtest ---
if data is not None:
    st.success("✅ Données chargées avec succès")

    if strategy_name == "Bandes de Bollinger":
        class CustomBollingerStrategy(BollingerBandsStrategy):
            n_period = bollinger_period
            n_std = bollinger_std
        StrategyToTest = CustomBollingerStrategy
    elif strategy_name == "Moyenne Mobile":
        StrategyToTest = MovingAverageCrossStrategy
    elif strategy_name == "RSI":
        StrategyToTest = RSIStrategy

    bt = Backtest(data, StrategyToTest, cash=initial_cash, commission=0.002)

    try:
        if strategy_name == "Bandes de Bollinger" and optimize:
            stats = bt.optimize(
                n_period=range(10, 40, 5),
                n_std=[1.5, 2.0, 2.5, 3.0],
                maximize='Sharpe Ratio',
                constraint=lambda p: p.n_period < len(data) / 2
            )
        else:
            stats = bt.run()

        # --- Affichage des résultats ---
        st.header("📊 Résultats du Backtest")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rendement stratégie", f"{stats['Return [%]']:.2f}%", delta=f"{stats['Return [%]'] - stats['Buy & Hold Return [%]']:.2f}% vs B&H")
            st.metric("B&H", f"{stats['Buy & Hold Return [%]']:.2f}%")

        with col2:
            st.metric("Max Drawdown", f"{stats['Max. Drawdown [%]']:.2f}%")
            st.metric("Win Rate", f"{stats['Win Rate [%]']:.2f}%")

        with col3:
            st.metric("Nbre Transactions", stats['# Trades'])
            st.metric("Gain/trade", f"{stats['Avg. Trade [%]']:.2f}%")

        with col4:
            st.metric("Capital Final", f"{currency_symbol}{stats['Equity Final [$]']:,.2f}")
            st.metric("Durée", str(stats['Duration']))

        st.markdown("---")
        st.subheader("🧠 Interprétation Automatique")
        sharpe = stats['Sharpe Ratio']
        if sharpe < 0.5:
            st.warning("⚠️ Rapport rendement/risque faible.")
        elif sharpe < 1:
            st.info("ℹ️ Stratégie modérément efficace.")
        else:
            st.success("✅ Bonne stratégie avec bon Sharpe Ratio.")

        # --- Graphique et export ---
        st.subheader("📈 Graphique")
        st.plotly_chart(bt.plot(), use_container_width=True)

        with st.expander("📋 Détails statistiques"):
            st.dataframe(stats)

        # --- Export des résultats ---
        csv = stats.to_frame().T.to_csv(index=False).encode()
        st.download_button("📥 Télécharger résultats CSV", data=csv, file_name="backtest_stats.csv", mime='text/csv')

    except Exception as e:
        st.error(f"Erreur durant le backtest : {e}")
else:
    st.info("Veuillez téléverser un fichier CSV pour commencer.")
