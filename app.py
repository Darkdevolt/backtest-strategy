import streamlit as st
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# --- Configuration de la page Streamlit ---
st.set_page_config(page_title="Backtesting Bandes de Bollinger", layout="wide")

st.title("Backtesting d'une Stratégie sur les Bandes de Bollinger")
st.write("Téléversez un fichier CSV avec des données historiques pour commencer.")

# --- Panneau latéral pour les paramètres ---
with st.sidebar:
    st.header("1. Vos Données")
    uploaded_file = st.file_uploader(
        "Téléversez votre fichier CSV",
        type="csv"
    )

    st.header("2. Paramètres de la Stratégie")
    bollinger_period = st.slider("Période des Bandes de Bollinger", 5, 100, 20)
    bollinger_std = st.slider("Écart-type des Bandes de Bollinger", 1.0, 4.0, 2.0, 0.1)

# --- Chargement et préparation des données ---
@st.cache_data
def load_uploaded_data(file):
    """Charge et prépare les données depuis un fichier CSV téléversé."""
    if file is None:
        return None
    
    try:
        df = pd.read_csv(file)
        
        # Nettoyage des noms de colonnes (suppression des espaces)
        df.columns = df.columns.str.strip()
        
        # Capitalisation pour correspondre au format standard (Open, High, Low, Close)
        required_columns = {'Date', 'Open', 'High', 'Low', 'Close'}
        df.columns = [col.capitalize() for col in df.columns]

        if not required_columns.issubset(df.columns):
            st.error(f"Le fichier CSV doit contenir les colonnes suivantes : {', '.join(required_columns)}")
            return None

        # Conversion de la colonne 'Date' et définition comme index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Tri des données par date
        df.sort_index(ascending=True, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la lecture du fichier : {e}")
        return None

data = load_uploaded_data(uploaded_file)

if data is not None:
    st.success("Fichier CSV chargé avec succès !")
    
    # --- Définition des indicateurs ---
    def bollinger_bands_indicator(series, n, std):
        """Calcule les bandes de Bollinger."""
        sma = series.rolling(n).mean()
        std_dev = series.rolling(n).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return sma, upper_band, lower_band

    # --- Stratégie de Backtesting ---
    class BollingerBandsStrategy(Strategy):
        n_period = bollinger_period
        n_std = bollinger_std

        def init(self):
            price = self.data.Close
            self.sma, self.upper, self.lower = self.I(
                bollinger_bands_indicator,
                price,
                self.n_period,
                self.n_std
            )

        def next(self):
            if crossover(self.data.Close, self.lower):
                self.buy()
            elif crossover(self.data.Close, self.upper):
                if self.position:
                    self.position.close()

    # --- Exécution du Backtest ---
    bt = Backtest(data, BollingerBandsStrategy, cash=10000, commission=.002)
    stats = bt.run()

    # --- Affichage des résultats ---
    st.header("Résultats du Backtesting")
    st.table(stats)

    st.header("Graphique de la Stratégie")
    fig = bt.plot()
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Veuillez téléverser un fichier CSV pour lancer le backtest.")
