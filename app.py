import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(page_title="SOGc Trading Strategy", layout="wide")
st.title("🚀 SOGc - Trend Momentum System")
st.write("""
**Stratégie quantitative pour battre le Buy & Hold**  
Combinaison de Moyennes Mobiles, Momentum et Gestion des Risques
""")

# Fonctions de calcul des indicateurs
def calculate_indicators(df):
    # Normaliser les noms de colonnes
    df.columns = df.columns.str.strip().str.lower()
    
    # Vérifier les colonnes nécessaires
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Colonne manquante: {col}. Les colonnes disponibles sont: {list(df.columns)}")
            return None

    # Moyennes Mobiles Exponentielles
    df['mme20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['mme100'] = df['close'].ewm(span=100, adjust=False).mean()
    
    # Rate of Change (14 jours)
    df['roc14'] = df['close'].pct_change(periods=14) * 100
    
    # Average True Range (14 jours)
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift())
    df['l-pc'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    df['atr14'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    
    return df.dropna()

# Génération des signaux
def generate_signals(df):
    df['signal'] = 0
    
    # Conditions d'achat
    buy_condition = (df['mme20'] > df['mme100']) & (df['roc14'] > 0) & (df['close'] > df['mme20'])
    df.loc[buy_condition, 'signal'] = 1
    
    # Conditions de vente
    sell_condition = (df['close'] < df['mme20']) | (df['roc14'] < -5)
    df.loc[sell_condition, 'signal'] = -1
    
    return df

# Backtesting de la stratégie
def backtest_strategy(df, initial_capital=1000000):
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    in_position = False
    
    for i, row in df.iterrows():
        if not in_position and row['signal'] == 1:
            # Entrée en position
            position = capital // row['close']
            entry_price = row['close']
            stop_loss = entry_price - (2 * row['atr14'])
            take_profit = entry_price + (6 * row['atr14'])
            capital -= position * entry_price
            in_position = True
            trades.append({
                'date': i,
                'type': 'BUY',
                'price': entry_price,
                'shares': position,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
        
        elif in_position:
            # Vérifier stop-loss et take-profit
            if row['low'] <= stop_loss:
                # Sortie par stop-loss
                capital += position * stop_loss
                trades.append({
                    'date': i,
                    'type': 'SELL',
                    'price': stop_loss,
                    'shares': position,
                    'reason': 'Stop-Loss'
                })
                in_position = False
                position = 0
            
            elif row['high'] >= take_profit:
                # Sortie par take-profit
                capital += position * take_profit
                trades.append({
                    'date': i,
                    'type': 'SELL',
                    'price': take_profit,
                    'shares': position,
                    'reason': 'Take-Profit'
                })
                in_position = False
                position = 0
            
            elif row['signal'] == -1:
                # Sortie par signal
                capital += position * row['close']
                trades.append({
                    'date': i,
                    'type': 'SELL',
                    'price': row['close'],
                    'shares': position,
                    'reason': 'Signal'
                })
                in_position = False
                position = 0
    
    # Calcul des métriques de performance
    if trades:
        df_trades = pd.DataFrame(trades)
        df_trades['profit'] = 0
        for i in range(1, len(df_trades)):
            if df_trades.iloc[i]['type'] == 'SELL':
                buy_price = df_trades.iloc[i-1]['price']
                sell_price = df_trades.iloc[i]['price']
                shares = df_trades.iloc[i-1]['shares']
                df_trades.at[i, 'profit'] = (sell_price - buy_price) * shares
        
        df_trades['cum_profit'] = df_trades['profit'].cumsum() + initial_capital
        win_trades = df_trades[df_trades['profit'] > 0]
        win_rate = len(win_trades) / len(df_trades[df_trades['type'] == 'SELL']) if len(df_trades[df_trades['type'] == 'SELL']) > 0 else 0
    else:
        df_trades = pd.DataFrame()
        win_rate = 0
    
    return capital, df_trades, win_rate

# Fonction principale
def main():
    # Télécharger les données
    uploaded_file = st.file_uploader("Télécharger les données historiques (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Afficher un aperçu des données
            st.subheader("Aperçu des données")
            st.write(f"**Période:** {df.index[0].date()} au {df.index[-1].date()}")
            st.write(f"**Colonnes:** {list(df.columns)}")
            st.dataframe(df.head(3))
            
            # Calcul des indicateurs
            df = calculate_indicators(df)
            if df is None:
                return
                
            # Génération des signaux
            df = generate_signals(df)
            
            # Backtesting
            initial_capital = 1000000
            final_capital, trades, win_rate = backtest_strategy(df.copy(), initial_capital)
            
            # Calcul du Buy & Hold
            buy_hold = initial_capital * (df['close'][-1] / df['close'][0])
            
            # Affichage des résultats
            st.subheader("📊 Résultats de la stratégie")
            col1, col2, col3 = st.columns(3)
            col1.metric("Stratégie Finale", f"{final_capital:,.0f} XOF", 
                       f"{(final_capital - initial_capital)/initial_capital:.2%}")
            col2.metric("Buy & Hold", f"{buy_hold:,.0f} XOF", 
                       f"{(buy_hold - initial_capital)/initial_capital:.2%}")
            col3.metric("Performance Relative", f"{(final_capital - buy_hold)/buy_hold:.2%}", 
                       f"{final_capital - buy_hold:,.0f} XOF")
            
            st.write(f"**Taux de réussite:** {win_rate:.2%} | **Nombre de trades:** {len(trades)//2 if not trades.empty else 0}")
            
            # Graphique principal
            st.subheader("📈 Graphique des prix et signaux")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Prix et moyennes mobiles
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Prix', line=dict(color='#1f77b4')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['mme20'], name='MME20', line=dict(color='#ff7f0e', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['mme100'], name='MME100', line=dict(color='#2ca02c', width=2)), row=1, col=1)
            
            # Signaux d'achat/vente
            buy_signals = df[df['signal'] == 1]
            sell_signals = df[df['signal'] == -1]
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'], 
                                        name='Achat', mode='markers', marker=dict(symbol='triangle-up', size=10, color='green')),
                            row=1, col=1)
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'], 
                                        name='Vente', mode='markers', marker=dict(symbol='triangle-down', size=10, color='red')),
                            row=1, col=1)
            
            # ROC
            fig.add_trace(go.Scatter(x=df.index, y=df['roc14'], name='ROC14', line=dict(color='#9467bd')), row=2, col=1)
            fig.add_hline(y=0, line=dict(color='gray', dash='dash'), row=2, col=1)
            
            # Mise en forme
            fig.update_layout(height=800, title="Prix et Signaux de Trading")
            fig.update_yaxes(title_text="Prix (XOF)", row=1, col=1)
            fig.update_yaxes(title_text="ROC14 (%)", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)
            
            # Graphique de performance
            if not trades.empty:
                st.subheader("📈 Performance comparée")
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Scatter(x=df.index, y=df['close']/df['close'][0]*initial_capital, 
                                            name='Buy & Hold', line=dict(color='#888')))
                fig_perf.add_trace(go.Scatter(x=trades[trades['type']=='SELL']['date'], 
                                            y=trades[trades['type']=='SELL']['cum_profit'],
                                            name='Stratégie', line=dict(color='#2ca02c', width=3)))
                
                fig_perf.update_layout(title="Performance Comparée",
                                    yaxis_title="Valeur du Portefeuille (XOF)",
                                    height=500)
                st.plotly_chart(fig_perf, use_container_width=True)
            
            # Détails des trades
            if not trades.empty:
                st.subheader("📋 Détails des trades")
                st.dataframe(trades)
            
            # Téléchargement des résultats
            if not trades.empty:
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Télécharger les résultats",
                    data=csv,
                    file_name="trading_results.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Une erreur s'est produite: {str(e)}")
            st.error("Veuillez vérifier le format de votre fichier de données")
    else:
        st.info("ℹ️ Veuillez télécharger un fichier CSV de données historiques pour commencer l'analyse")
        st.info("Le fichier doit contenir les colonnes: date, open, high, low, close, volume")

if __name__ == "__main__":
    main()
