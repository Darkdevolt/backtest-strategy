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

# --- Fonction de chargement de données améliorée ---
@st.cache_data
def load_uploaded_data(file):
    """Charge et prépare les données depuis un fichier CSV téléversé."""
    if file is None:
        return None
    
    try:
        df = pd.read_csv(file)
        
        # Nettoyage des noms de colonnes (suppression des espaces)
        df.columns = df.columns.str.strip()
        
        # Capitalisation pour correspondre au format standard
        df.columns = [col.capitalize() for col in df.columns]
        
        required_columns = {'Date', 'Open', 'High', 'Low', 'Close'}
        if not required_columns.issubset(df.columns):
            st.error(f"Le fichier CSV doit contenir les colonnes suivantes : {', '.join(required_columns)}")
            return None

        # Conversion et validation des types de données
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'Volume' in df.columns:
            df[col] = pd.to_numeric(df['Volume'], errors='coerce')

        df.set_index('Date', inplace=True)
        df.dropna(inplace=True) # Supprimer les lignes avec des données manquantes
        df.sort_index(ascending=True, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la lecture du fichier : {e}")
        return None

data = load_uploaded_data(uploaded_file)

# --- Exécution du backtest uniquement si les données sont valides ---
if data is not None:
    # ❗️ NOUVELLE VÉRIFICATION ICI ❗️
    # Vérifie si le nombre de points de données est suffisant
    if len(data) <= bollinger_period:
        st.error(
            f"Erreur : La période des Bandes de Bollinger ({bollinger_period}) est plus grande "
            f"ou égale au nombre de points de données disponibles ({len(data)}).\n\n"
            "Veuillez choisir une période plus courte ou téléverser un fichier avec plus de données."
        )
    else:
        st.success("Fichier CSV chargé avec succès et prêt pour l'analyse !")
        
        # --- Définition de l'indicateur ---
        def bollinger_bands_indicator(series, n, std):
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
                # L'indicateur est appelé ici
                self.sma, self.upper, self.lower = self.I(
                    bollinger_bands_indicator,
                    pd.Series(price), # Assurer que c'est bien une série Pandas
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
        try:
            bt = Backtest(data, BollingerBandsStrategy, cash=10000, commission=.002)
            stats = bt.run()

            # --- Affichage des résultats ---
            st.header("Résultats du Backtesting")
            st.table(stats)

            st.header("Graphique de la Stratégie")
            fig = bt.plot()
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Une erreur est survenue pendant le backtesting : {e}")

else:
    st.info("Veuillez téléverser un fichier CSV pour lancer le backtest.")
