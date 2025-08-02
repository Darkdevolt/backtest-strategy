import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(page_title="SOGc Trading Strategy", layout="wide")
st.title("🚀 SOGc - Système de Trading Quantitatif")
st.write("""
**Stratégies quantitatives avancées avec gestion personnalisée du risque**  
Choisissez votre stratégie et paramétrez vos règles de trading
""")

# Fonctions de calcul des indicateurs
def calculate_indicators(df, strategy_params):
    # Normaliser les noms de colonnes
    df.columns = df.columns.str.strip().str.lower()
    
    # Vérifier les colonnes nécessaires
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Colonne manquante: {col}. Les colonnes disponibles sont: {list(df.columns)}")
            return None

    # Calculs communs à toutes les stratégies
    # Average True Range (ATR)
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift())
    df['l-pc'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    df['atr'] = df['tr'].ewm(alpha=1/strategy_params['atr_period'], adjust=False).mean()
    
    # Calculs spécifiques aux stratégies
    if strategy_params['strategy'] == "Trend Momentum":
        # Moyennes Mobiles Exponentielles
        df['mme_fast'] = df['close'].ewm(span=strategy_params['fast_ma'], adjust=False).mean()
        df['mme_slow'] = df['close'].ewm(span=strategy_params['slow_ma'], adjust=False).mean()
        
        # Rate of Change
        df['roc'] = df['close'].pct_change(periods=strategy_params['roc_period']) * 100
    
    elif strategy_params['strategy'] == "Bollinger Bands":
        # Moyenne Mobile Simple
        df['sma'] = df['close'].rolling(window=strategy_params['bb_period']).mean()
        
        # Écart-type
        df['std'] = df['close'].rolling(window=strategy_params['bb_period']).std()
        
        # Bandes de Bollinger
        df['bb_upper'] = df['sma'] + (strategy_params['bb_std'] * df['std'])
        df['bb_lower'] = df['sma'] - (strategy_params['bb_std'] * df['std'])
        
        # Bandwidth et B%
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['sma']
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df.dropna()

# Génération des signaux
def generate_signals(df, strategy_params):
    df['signal'] = 0
    
    if strategy_params['strategy'] == "Trend Momentum":
        # Conditions d'achat
        buy_condition = (
            (df['mme_fast'] > df['mme_slow']) & 
            (df['roc'] > strategy_params['roc_threshold']) & 
            (df['close'] > df['mme_fast'])
        )
        
        # Conditions de vente
        sell_condition = (
            (df['close'] < df['mme_fast']) | 
            (df['roc'] < strategy_params['roc_exit'])
        )
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
    
    elif strategy_params['strategy'] == "Bollinger Bands":
        # Conditions d'achat (survente)
        buy_condition = (
            (df['close'] < df['bb_lower']) & 
            (df['bb_percent'] < strategy_params['bb_buy_threshold'])
        )
        
        # Conditions de vente (surachat ou retour à la moyenne)
        sell_condition = (
            (df['close'] > df['bb_upper']) | 
            (df['close'] > df['sma'] + strategy_params['bb_exit_offset'] * df['atr'])
        )
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
    
    return df

# Backtesting de la stratégie
def backtest_strategy(df, initial_capital, risk_per_trade, strategy_params):
    cash = initial_capital
    positions = 0
    entry_price = 0
    trades = []
    in_position = False
    
    for i, row in df.iterrows():
        current_price = row['close']
        
        if not in_position and row['signal'] == 1:
            # Calcul du capital à risquer
            risk_capital = cash * risk_per_trade
            
            # Vérifier qu'on a assez de capital
            if risk_capital < current_price:
                continue
                
            # Entrée en position
            position = int(risk_capital // current_price)
            entry_price = current_price
            
            # Définir stop-loss et take-profit
            if strategy_params['sl_type'] == "ATR":
                stop_loss = entry_price - (strategy_params['sl_atr_mult'] * row['atr'])
            else:
                stop_loss = entry_price * (1 - strategy_params['sl_percent']/100)
                
            if strategy_params['tp_type'] == "ATR":
                take_profit = entry_price + (strategy_params['tp_atr_mult'] * row['atr'])
            else:
                take_profit = entry_price * (1 + strategy_params['tp_percent']/100)
            
            # Mettre à jour le cash et les positions
            cash -= position * entry_price
            positions = position
            in_position = True
            
            trades.append({
                'date': i,
                'type': 'BUY',
                'price': entry_price,
                'shares': position,
                'amount': position * entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'remaining_cash': cash
            })
        
        elif in_position:
            # Vérifier stop-loss et take-profit
            exit_reason = None
            exit_price = current_price
            
            if row['low'] <= stop_loss:
                exit_price = stop_loss
                exit_reason = 'Stop-Loss'
            elif row['high'] >= take_profit:
                exit_price = take_profit
                exit_reason = 'Take-Profit'
            elif row['signal'] == -1:
                exit_reason = 'Signal'
            
            if exit_reason:
                # Sortie de position
                cash += positions * exit_price
                
                trades.append({
                    'date': i,
                    'type': 'SELL',
                    'price': exit_price,
                    'shares': positions,
                    'amount': positions * exit_price,
                    'reason': exit_reason,
                    'remaining_cash': cash
                })
                
                in_position = False
                positions = 0
                entry_price = 0
    
    # Calcul du capital final
    final_capital = cash + (positions * df['close'].iloc[-1] if in_position else cash)
    
    # Calcul des métriques de performance
    if trades:
        df_trades = pd.DataFrame(trades)
        
        # Calculer les profits
        df_trades['trade_profit'] = 0
        df_trades['cumulative_profit'] = 0
        cumulative_profit = 0
        
        for i in range(len(df_trades)):
            if df_trades.iloc[i]['type'] == 'SELL':
                # Trouver le trade d'achat précédent
                buy_index = df_trades[df_trades.index < i][df_trades['type'] == 'BUY'].index[-1]
                buy_price = df_trades.loc[buy_index]['price']
                sell_price = df_trades.iloc[i]['price']
                shares = df_trades.loc[buy_index]['shares']
                
                profit = (sell_price - buy_price) * shares
                df_trades.at[i, 'trade_profit'] = profit
                
                cumulative_profit += profit
                df_trades.at[i, 'cumulative_profit'] = cumulative_profit
        
        # Taux de réussite
        sell_trades = df_trades[df_trades['type'] == 'SELL']
        if not sell_trades.empty:
            winning_trades = sell_trades[sell_trades['trade_profit'] > 0]
            win_rate = len(winning_trades) / len(sell_trades)
        else:
            win_rate = 0
    else:
        df_trades = pd.DataFrame()
        win_rate = 0
    
    return final_capital, df_trades, win_rate

# Fonction principale
def main():
    # Télécharger les données
    uploaded_file = st.file_uploader("Télécharger les données historiques (CSV)", type="csv")
    
    # Initialisation des paramètres par défaut
    strategy_params = {
        'strategy': "Trend Momentum",
        'atr_period': 14,
        'fast_ma': 20,
        'slow_ma': 100,
        'roc_period': 14,
        'roc_threshold': 0,
        'roc_exit': -5,
        'bb_period': 20,
        'bb_std': 2,
        'bb_buy_threshold': 0.1,
        'bb_exit_offset': 0.5,
        'sl_type': "ATR",
        'sl_percent': 5,
        'sl_atr_mult': 2,
        'tp_type': "ATR",
        'tp_percent': 10,
        'tp_atr_mult': 6
    }
    
    # Paramètres de trading
    with st.sidebar:
        st.header("⚙️ Paramètres de Trading")
        
        # Capital et risque
        initial_capital = st.number_input("Capital Initial (XOF)", 
                                         min_value=10000, 
                                         value=1000000, 
                                         step=10000)
        
        risk_per_trade = st.slider("% du Capital Risqué par Trade", 
                                 min_value=0.1, 
                                 max_value=100.0, 
                                 value=2.0, 
                                 step=0.5) / 100.0
        
        st.info(f"Montant risqué par trade: {initial_capital * risk_per_trade:,.0f} XOF")
        
        # Sélection de la stratégie
        st.subheader("📊 Stratégie de Trading")
        strategy = st.selectbox("Choisissez une stratégie", 
                               ["Trend Momentum", "Bollinger Bands"])
        strategy_params['strategy'] = strategy
        
        # Paramètres ATR communs
        st.subheader("📈 Paramètres Communs")
        strategy_params['atr_period'] = st.slider("Période ATR", 5, 50, 14)
        
        # Paramètres spécifiques à la stratégie
        st.subheader("⚡ Paramètres Spécifiques")
        
        if strategy == "Trend Momentum":
            strategy_params['fast_ma'] = st.slider("MME Rapide", 5, 50, 20)
            strategy_params['slow_ma'] = st.slider("MME Lente", 50, 200, 100)
            strategy_params['roc_period'] = st.slider("Période ROC", 5, 50, 14)
            strategy_params['roc_threshold'] = st.slider("Seuil ROC Entrée", -5.0, 10.0, 0.0, 0.5)
            strategy_params['roc_exit'] = st.slider("Seuil ROC Sortie", -20.0, 5.0, -5.0, 0.5)
        
        elif strategy == "Bollinger Bands":
            strategy_params['bb_period'] = st.slider("Période Bollinger", 10, 50, 20)
            strategy_params['bb_std'] = st.slider("Déviation Standard", 1.0, 3.0, 2.0, 0.1)
            strategy_params['bb_buy_threshold'] = st.slider("Seuil d'Achat (B%)", 0.0, 1.0, 0.1, 0.05)
            strategy_params['bb_exit_offset'] = st.slider("Décalage Sortie (ATR)", 0.1, 2.0, 0.5, 0.1)
        
        # Paramètres de gestion du risque
        st.subheader("⚠️ Paramètres de Risque")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Stop-Loss**")
            strategy_params['sl_type'] = st.radio("Type SL", ["ATR", "Pourcentage"], index=0)
            if strategy_params['sl_type'] == "ATR":
                strategy_params['sl_atr_mult'] = st.slider("Multiplicateur ATR SL", 0.5, 5.0, 2.0, 0.5)
            else:
                strategy_params['sl_percent'] = st.slider("% Stop-Loss", 1.0, 20.0, 5.0, 0.5)
        
        with col2:
            st.write("**Take-Profit**")
            strategy_params['tp_type'] = st.radio("Type TP", ["ATR", "Pourcentage"], index=0)
            if strategy_params['tp_type'] == "ATR":
                strategy_params['tp_atr_mult'] = st.slider("Multiplicateur ATR TP", 1.0, 10.0, 6.0, 0.5)
            else:
                strategy_params['tp_percent'] = st.slider("% Take-Profit", 1.0, 50.0, 10.0, 0.5)
    
    if uploaded_file is not None:
        try:
            # Chargement des données
            df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Afficher un aperçu des données
            st.subheader("📊 Aperçu des données")
            st.write(f"**Période:** {df.index[0].date()} au {df.index[-1].date()}")
            st.write(f"**Colonnes:** {list(df.columns)}")
            st.dataframe(df.head(3))
            
            # Calcul des indicateurs
            df = calculate_indicators(df, strategy_params)
            if df is None:
                return
                
            # Génération des signaux
            df = generate_signals(df, strategy_params)
            
            # Backtesting
            final_capital, trades, win_rate = backtest_strategy(
                df.copy(), 
                initial_capital, 
                risk_per_trade,
                strategy_params
            )
            
            # Calcul du Buy & Hold
            buy_hold = initial_capital * (df['close'].iloc[-1] / df['close'].iloc[0])
            
            # Affichage des résultats
            st.subheader("📈 Résultats de la stratégie")
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
            
            # Prix
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Prix', line=dict(color='#1f77b4')), row=1, col=1)
            
            # Indicateurs spécifiques
            if strategy_params['strategy'] == "Trend Momentum":
                fig.add_trace(go.Scatter(x=df.index, y=df['mme_fast'], name='MME Rapide', line=dict(color='#ff7f0e', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['mme_slow'], name='MME Lente', line=dict(color='#2ca02c', width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['roc'], name='ROC', line=dict(color='#9467bd')), row=2, col=1)
                fig.add_hline(y=0, line=dict(color='gray', dash='dash'), row=2, col=1)
                
            elif strategy_params['strategy'] == "Bollinger Bands":
                fig.add_trace(go.Scatter(x=df.index, y=df['sma'], name='SMA', line=dict(color='#2ca02c', width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='BB Sup', line=dict(color='#d62728', width=1, dash='dot')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='BB Inf', line=dict(color='#1f77b4', width=1, dash='dot')), row=1, col=1)
                
                # Remplissage entre les bandes - CORRIGÉ
                fig.add_trace(go.Scatter(
                    x=np.concatenate([df.index, df.index[::-1]]),
                    y=np.concatenate([df['bb_upper'], df['bb_lower'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(44, 160, 44, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Bollinger Bands',
                    showlegend=False
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df.index, y=df['bb_percent'], name='B%', line=dict(color='#9467bd')), row=2, col=1)
                fig.add_hline(y=0.2, line=dict(color='green', dash='dash'), row=2, col=1)
                fig.add_hline(y=0.8, line=dict(color='red', dash='dash'), row=2, col=1)
            
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
            
            # Mise en forme
            fig.update_layout(height=800, title=f"Prix et Signaux - Stratégie {strategy_params['strategy']}")
            fig.update_yaxes(title_text="Prix (XOF)", row=1, col=1)
            fig.update_yaxes(title_text="Indicateur", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)
            
            # Graphique de performance
            if not trades.empty:
                st.subheader("📊 Performance comparée")
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Scatter(x=df.index, y=df['close']/df['close'].iloc[0]*initial_capital, 
                                            name='Buy & Hold', line=dict(color='#888')))
                
                # Calculer l'évolution du capital pour la stratégie
                strategy_value = [initial_capital]
                strategy_dates = [df.index[0]]
                cash = initial_capital
                
                for trade in trades.to_dict('records'):
                    if trade['type'] == 'BUY':
                        cash = trade['remaining_cash']
                        strategy_value.append(cash)
                        strategy_dates.append(trade['date'])
                    elif trade['type'] == 'SELL':
                        cash = trade['remaining_cash']
                        strategy_value.append(cash)
                        strategy_dates.append(trade['date'])
                
                # Ajouter le dernier point si nécessaire
                if strategy_dates[-1] != df.index[-1]:
                    strategy_dates.append(df.index[-1])
                    strategy_value.append(cash)
                
                fig_perf.add_trace(go.Scatter(x=strategy_dates, 
                                            y=strategy_value,
                                            name='Stratégie', line=dict(color='#2ca02c', width=3)))
                
                fig_perf.update_layout(title="Évolution du Capital",
                                    yaxis_title="Valeur du Portefeuille (XOF)",
                                    height=500)
                st.plotly_chart(fig_perf, use_container_width=True)
            
            # Détails des trades
            if not trades.empty:
                st.subheader("📋 Détails des trades")
                
                # Calculer le profit par trade
                trades['profit'] = 0
                trades['cumulative_profit'] = 0
                cumulative_profit = 0
                
                for i in range(len(trades)):
                    if trades.iloc[i]['type'] == 'SELL':
                        # Trouver le trade d'achat précédent
                        buy_index = trades[trades.index < i][trades['type'] == 'BUY'].index[-1]
                        buy_price = trades.loc[buy_index]['price']
                        sell_price = trades.iloc[i]['price']
                        shares = trades.loc[buy_index]['shares']
                        
                        profit = (sell_price - buy_price) * shares
                        trades.at[i, 'profit'] = profit
                        
                        cumulative_profit += profit
                        trades.at[i, 'cumulative_profit'] = cumulative_profit
                
                # Formater les colonnes
                trades_display = trades.copy()
                numeric_cols = ['price', 'amount', 'stop_loss', 'take_profit', 'remaining_cash', 'profit', 'cumulative_profit']
                
                for col in numeric_cols:
                    if col in trades_display:
                        trades_display[col] = trades_display[col].apply(lambda x: f"{x:,.0f} XOF" if not pd.isna(x) else "")
                
                st.dataframe(trades_display)
            
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
