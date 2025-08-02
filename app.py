import streamlit as st
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# --- Configuration de la page Streamlit ---
st.set_page_config(page_title="Backtesting Bandes de Bollinger", layout="wide")

st.title("📈 Backtesting d'une Stratégie sur les Bandes de Bollinger")
st.write("Téléversez vos données et ajustez les paramètres pour visualiser les performances de la stratégie.")

# --- Panneau latéral pour les paramètres ---
with st.sidebar:
    st.header("1. Vos Données")
    uploaded_file = st.file_uploader(
        "Téléversez votre fichier CSV",
        type="csv"
    )

    st.header("2. Paramètres du Portefeuille")
    initial_cash = st.number_input("Capital de départ", min_value=100, value=10000, step=100)
    currency_symbol = st.text_input("Symbole de la Devise", value="$")


    st.header("3. Paramètres de la Stratégie")
    bollinger_period = st.slider("Période des Bandes de Bollinger", 5, 100, 20)
    bollinger_std = st.slider("Écart-type des Bandes de Bollinger", 1.0, 4.0, 2.0, 0.1)

# --- Fonction de chargement de données ---
@st.cache_data
def load_uploaded_data(file):
    if file is None: return None
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df.columns = [col.capitalize() for col in df.columns]
        required_columns = {'Date', 'Open', 'High', 'Low', 'Close'}
        if not required_columns.issubset(df.columns):
            st.error(f"Le fichier CSV doit contenir les colonnes suivantes : {', '.join(required_columns)}")
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df.set_index('Date', inplace=True)
        df.dropna(inplace=True)
        df.sort_index(ascending=True, inplace=True)
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return None

data = load_uploaded_data(uploaded_file)

if data is not None:
    if len(data) <= bollinger_period:
        st.error(
            f"Erreur : La période des Bandes de Bollinger ({bollinger_period}) est plus grande que le nombre de points de données ({len(data)}). "
            "Veuillez choisir une période plus courte."
        )
    else:
        st.success("Fichier CSV chargé avec succès !")

        class BollingerBandsStrategy(Strategy):
            n_period = bollinger_period
            n_std = bollinger_std
            def init(self):
                price = self.data.Close
                self.sma, self.upper, self.lower = self.I(
                    lambda s, n, std: (s.rolling(n).mean(), s.rolling(n).mean() + std * s.rolling(n).std(), s.rolling(n).mean() - std * s.rolling(n).std()),
                    pd.Series(price), self.n_period, self.n_std
                )
            def next(self):
                if crossover(self.data.Close, self.lower):
                    self.buy()
                elif crossover(self.data.Close, self.upper):
                    if self.position: self.position.close()
        
        try:
            # --- Utilisation du capital initial personnalisé ---
            bt = Backtest(data, BollingerBandsStrategy, cash=initial_cash, commission=.002)
            stats = bt.run()

            st.header("📊 Tableau de Bord des Résultats")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(label="Rendement Stratégie", value=f"{stats['Return [%]']:.2f}%", delta=f"{stats['Return [%]'] - stats['Buy & Hold Return [%]']:.2f}% vs Buy & Hold")
                st.metric(label="Rendement Buy & Hold", value=f"{stats['Buy & Hold Return [%]']:.2f}%")

            with col2:
                st.metric(label="📉 Max Drawdown", value=f"{stats['Max. Drawdown [%]']:.2f}%")
                st.metric(label="📈 Taux de Réussite (Win Rate)", value=f"{stats['Win Rate [%]']:.2f}%")
            
            with col3:
                st.metric(label="Nombre de Transactions", value=stats['# Trades'])
                st.metric(label="Gain/Perte Moyen par Trade", value=f"{stats['Avg. Trade [%]']:.2f}%")

            with col4:
                 # --- Utilisation du symbole de devise personnalisé ---
                 st.metric(label="Capital Final", value=f"{currency_symbol}{stats['Equity Final [$]']:,.2f}")
                 st.metric(label="Durée de la Simulation", value=f"{stats['Duration']}")

            st.markdown("---")
            st.header("CHARTS")
            fig = bt.plot()
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Voir toutes les statistiques détaillées"):
                st.table(stats)

        except Exception as e:
            st.error(f"Une erreur est survenue pendant le backtesting : {e}")
else:
    st.info("Veuillez téléverser un fichier CSV pour lancer le backtest.")
