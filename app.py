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
    # Moyennes Mobiles Exponentielles
    df['MME20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['MME100'] = df['Close'].ewm(span=100, adjust=False).mean()
    
    # Rate of Change (14 jours)
    df['ROC14'] = df['Close'].pct_change(periods=14) * 100
    
    # Average True Range (14 jours)
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift())
    df['L-PC'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR14'] = df['TR'].ewm(alpha=1/14, adjust=False).mean()
    
    return df.dropna()

# Génération des signaux
def generate_signals(df):
    df['Signal'] = 0
    
    # Conditions d'achat
    buy_condition = (df['MME20'] > df['MME100']) & (df['ROC14'] > 0) & (df['Close'] > df['MME20'])
    df.loc[buy_condition, 'Signal'] = 1
    
    # Conditions de vente
    sell_condition = (df['Close'] < df['MME20']) | (df['ROC14'] < -5)
    df.loc[sell_condition, 'Signal'] = -1
    
    return df

# Backtesting de la stratégie
def backtest_strategy(df, initial_capital=1000000):
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    in_position = False
    
    for i, row in df.iterrows():
        if not in_position and row['Signal'] == 1:
            # Entrée en position
            position = capital // row['Close']
            entry_price = row['Close']
            stop_loss = entry_price - (2 * row['ATR14'])
            take_profit = entry_price + (6 * row['ATR14'])
            capital -= position * entry_price
            in_position = True
            trades.append({
                'Date': i,
                'Type': 'BUY',
                'Price': entry_price,
                'Shares': position,
                'StopLoss': stop_loss,
                'TakeProfit': take_profit
            })
        
        elif in_position:
            # Vérifier stop-loss et take-profit
            if row['Low'] <= stop_loss:
                # Sortie par stop-loss
                capital += position * stop_loss
                trades.append({
                    'Date': i,
                    'Type': 'SELL',
                    'Price': stop_loss,
                    'Shares': position,
                    'Reason': 'Stop-Loss'
                })
                in_position = False
                position = 0
            
            elif row['High'] >= take_profit:
                # Sortie par take-profit
                capital += position * take_profit
                trades.append({
                    'Date': i,
                    'Type': 'SELL',
                    'Price': take_profit,
                    'Shares': position,
                    'Reason': 'Take-Profit'
                })
                in_position = False
                position = 0
            
            elif row['Signal'] == -1:
                # Sortie par signal
                capital += position * row['Close']
                trades.append({
                    'Date': i,
                    'Type': 'SELL',
                    'Price': row['Close'],
                    'Shares': position,
                    'Reason': 'Signal'
                })
                in_position = False
                position = 0
    
    # Calcul des métriques de performance
    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        df_trades['Profit'] = np.where(
            df_trades['Type'] == 'SELL',
            (df_trades['Price'] - df_trades['Price'].shift(1)) * df_trades['Shares'],
            0
        )
        df_trades['CumProfit'] = df_trades['Profit'].cumsum() + initial_capital
        win_rate = len(df_trades[df_trades['Profit'] > 0]) / len(df_trades[df_trades['Type'] == 'SELL'])
    else:
        win_rate = 0
        df_trades['CumProfit'] = initial_capital
    
    return capital, df_trades, win_rate

# Fonction principale
def main():
    # Télécharger les données
    uploaded_file = st.file_uploader("Télécharger les données historiques (CSV)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
        df.sort_index(inplace=True)
        
        # Calcul des indicateurs
        df = calculate_indicators(df)
        
        # Génération des signaux
        df = generate_signals(df)
        
        # Backtesting
        initial_capital = 1000000
        final_capital, trades, win_rate = backtest_strategy(df.copy(), initial_capital)
        
        # Calcul du Buy & Hold
        buy_hold = initial_capital * (df['Close'][-1] / df['Close'][0])
        
        # Affichage des résultats
        col1, col2, col3 = st.columns(3)
        col1.metric("Stratégie Finale", f"{final_capital:,.0f} XOF", 
                   f"{(final_capital - initial_capital)/initial_capital:.2%}")
        col2.metric("Buy & Hold", f"{buy_hold:,.0f} XOF", 
                   f"{(buy_hold - initial_capital)/initial_capital:.2%}")
        col3.metric("Performance Relative", f"{(final_capital - buy_hold)/buy_hold:.2%}", 
                   f"{final_capital - buy_hold:,.0f} XOF")
        
        st.write(f"**Taux de réussite:** {win_rate:.2%} | **Nombre de trades:** {len(trades)//2}")
        
        # Graphique principal
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Prix et moyennes mobiles
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Prix', line=dict(color='#1f77b4')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MME20'], name='MME20', line=dict(color='#ff7f0e', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MME100'], name='MME100', line=dict(color='#2ca02c', width=2)), row=1, col=1)
        
        # Signaux d'achat/vente
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], 
                                name='Achat', mode='markers', marker=dict(symbol='triangle-up', size=10, color='green')),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], 
                                name='Vente', mode='markers', marker=dict(symbol='triangle-down', size=10, color='red')),
                     row=1, col=1)
        
        # ROC
        fig.add_trace(go.Scatter(x=df.index, y=df['ROC14'], name='ROC14', line=dict(color='#9467bd')), row=2, col=1)
        fig.add_hline(y=0, line=dict(color='gray', dash='dash'), row=2, col=1)
        
        # Mise en forme
        fig.update_layout(height=800, title="Prix et Signaux de Trading")
        fig.update_yaxes(title_text="Prix (XOF)", row=1, col=1)
        fig.update_yaxes(title_text="ROC14 (%)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique de performance
        if not trades.empty:
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(x=df.index, y=df['Close']/df['Close'][0]*initial_capital, 
                                         name='Buy & Hold', line=dict(color='#888')))
            fig_perf.add_trace(go.Scatter(x=trades[trades['Type']=='SELL']['Date'], 
                                         y=trades[trades['Type']=='SELL']['CumProfit'],
                                         name='Stratégie', line=dict(color='#2ca02c', width=3)))
            
            fig_perf.update_layout(title="Performance Comparée",
                                  yaxis_title="Valeur du Portefeuille (XOF)",
                                  height=500)
            st.plotly_chart(fig_perf, use_container_width=True)
        
        # Détails des trades
        with st.expander("Voir les détails des trades"):
            if not trades.empty:
                st.dataframe(trades)
            else:
                st.write("Aucun trade effectué durant cette période")
        
        # Téléchargement des résultats
        csv = trades.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger les résultats",
            data=csv,
            file_name="trading_results.csv",
            mime="text/csv"
        )
    else:
        st.info("Veuillez télécharger un fichier CSV de données historiques pour commencer l'analyse")

if __name__ == "__main__":
    main()
