import pandas as pd
import numpy as np
import plotly.graph_objects as go

def fetch_data(excel_path, start_date=None, end_date=None):
    """
    Carica e pulisce i dati OHLC dal file Excel.
    Gestisce il riconoscimento automatico della colonna data, la conversione a datetime,
    e la validazione delle colonne necessarie.
    """
    try:
        df = pd.read_excel(excel_path)
        # Cerca una colonna con un nome comune per le date per renderla colonna indice
        date_column = next((col for col in ['Date', 'date', 'Datetime', 'datetime', 'TIME', 'Time', 'time'] if col in df.columns), None)
        if date_column is None: raise ValueError(f"Nessuna colonna data trovata in {df.columns}")
        
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True); df.sort_index(inplace=True)
        
        # Filtra per intervallo di date specificato
        if start_date and end_date: df = df.loc[start_date:end_date]
        
        # Controlla la presenza delle colonne OHLC e le converte in formato numerico
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns): raise ValueError(f"Colonne OHLC mancanti. Trovate: {df.columns}")
        for col in required_columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rimuove le righe con dati OHLC mancanti per garantire l'integrità del backtest
        df.dropna(subset=required_columns, inplace=True)
        
        if df.empty: return None
        return df
    except Exception as e:
        print(f"Errore durante il caricamento dei dati: {e}")
        return None

def add_indicators(df, moving_average_period=20, std_dev_multiplier=2.0):
    """
    Aggiunge al DataFrame gli indicatori tecnici necessari per la strategia, cioè le bande di bollinger.
    """
    if df is None: return None
    # Lavora su una copia per evitare modifiche impreviste sul dataframe originale
    df = df.copy()
    df['SMA'] = df['Close'].rolling(window=moving_average_period).mean()
    df['STD'] = df['Close'].rolling(window=moving_average_period).std()
    df['Upper'] = df['SMA'] + (df['STD'] * std_dev_multiplier)
    df['Lower'] = df['SMA'] - (df['STD'] * std_dev_multiplier)
    return df

def generate_signals_with_params(df, spread_pct=0.0, initial_capital=100000, risk_per_trade=1.0, commission_params=None):
    """
    Simula il backtest della strategia di trading.
    Itera attraverso le candele, genera segnali di entrata/uscita, calcola la dimensione 
    della posizione, gestisce i trade aperti e chiusi, e traccia l'equity.
    """
    if df is None or 'Upper' not in df.columns or 'Lower' not in df.columns:
        return None, initial_capital

    df = df.copy()
    
    # Inizializzazione delle colonne necessarie per tracciare lo stato del backtest e dei trade
    df['Signal'], df['Trade_Type'], df['Position_Size'], df['Entry_Price'], df['Exit_Price'] = [0, '', 0.0, np.nan, np.nan]
    df['Trade_Active'], df['Realized_PnL'], df['Unrealized_PnL'] = [False, 0.0, 0.0]
    df['Cash'], df['Equity'] = [float(initial_capital), float(initial_capital)]
    df['Swap_Cost'] = 0.0
    df['Exit_Type'] = ''
    # Colonne per avere i dettagli del trade al momento della chiusura
    df['Closed_Trade_Entry_Time'] = pd.NaT
    df['Closed_Trade_Entry_Price'] = np.nan
    df['Closed_Trade_Type'] = ''
    
    # Variabili per mantenere lo stato del trade attivo tra le varie iterazioni
    active_trade, trade_type, entry_price, position_size = False, '', 0.0, 0.0
    entry_time, last_swap_check_date = pd.NaT, None

    if commission_params is None: commission_params = {'model': 'none'}
    commission_model = commission_params.get('model', 'none')

    # Itera su ogni candela del DataFrame per simulare il trading
    for i in range(1, len(df)):
        current_index, prev_index = df.index[i], df.index[i-1]
        
        # Inizializza i valori della riga corrente con quelli della precedente (che verranno aggiornati quando accade qualcosa)
        df.loc[current_index, ['Cash', 'Equity']] = df.loc[prev_index, ['Cash', 'Equity']]
        
        prev_close, current_close = df.loc[prev_index, 'Close'], df.loc[current_index, 'Close']
        prev_upper, prev_lower = df.loc[prev_index, 'Upper'], df.loc[prev_index, 'Lower']
        current_upper, current_lower = df.loc[current_index, ['Upper', 'Lower']]

        # Salta l'iterazione se gli indicatori non sono ancora calcolati (all'inizio del DataFrame)
        if pd.isna(prev_upper) or pd.isna(prev_lower) or pd.isna(current_upper) or pd.isna(current_lower):
            continue

        # GESTIONE DI UN TRADE ATTIVO
        if active_trade:
            # 1. Calcolo dei costi overnight (Swap)
            days_open = (current_index - entry_time).days
            grace_period_days = commission_params.get('grace_period_days', 0)
            if commission_model != 'none' and days_open >= grace_period_days and (last_swap_check_date is None or current_index.date() > last_swap_check_date):
                swap_cost_today = 0.0; triple_fee_day = commission_params.get('triple_fee_day', 4); fee_multiplier = 3 if current_index.dayofweek == triple_fee_day else 1
                if commission_model == 'etoro_per_unit':
                    fee_per_unit = commission_params.get('long_fee_per_unit', 0.0) if trade_type == 'Long' else commission_params.get('short_fee_per_unit', 0.0); swap_cost_today = position_size * fee_per_unit * fee_multiplier
                if swap_cost_today != 0: df.loc[current_index, 'Swap_Cost'] = swap_cost_today; df.loc[current_index, 'Cash'] += swap_cost_today
                last_swap_check_date = current_index.date()
            
            # 2. Aggiornamento del P&L non realizzato (curva Mark-to-Market)
            unrealized_pnl = position_size * (current_close - entry_price) if trade_type == 'Long' else position_size * (entry_price - current_close)
            df.loc[current_index, 'Unrealized_PnL'] = unrealized_pnl; df.loc[current_index, 'Equity'] = df.loc[current_index, 'Cash'] + unrealized_pnl
        
        # Definizione delle condizioni di segnale: incrocio delle bande dal basso verso l'alto o viceversa
        is_buy_signal = prev_close < prev_lower and current_close > current_lower
        is_sell_signal = prev_close > prev_upper and current_close < current_upper

        # GESTIONE DELLE USCITE
        exit_signal_triggered = False
        # Logica di uscita "Stop-and-Reverse": se un trade è attivo e si verifica un segnale opposto, chiude il trade corrente
        if active_trade and ((trade_type == 'Long' and is_sell_signal) or (trade_type == 'Short' and is_buy_signal)):
            exit_signal_triggered = True
            spread_amount = current_close * spread_pct
            exit_price = current_close - spread_amount if trade_type == 'Long' else current_close + spread_amount
            pnl = position_size * (exit_price - entry_price) if trade_type == 'Long' else position_size * (entry_price - exit_price)
            # Registra i dettagli della chiusura del trade
            df.loc[current_index, ['Realized_PnL', 'Cash', 'Equity', 'Exit_Price', 'Exit_Type', 'Closed_Trade_Entry_Time', 'Closed_Trade_Entry_Price', 'Closed_Trade_Type']] = \
                [pnl, df.loc[current_index, 'Cash'] + pnl, df.loc[current_index, 'Cash'] + pnl, exit_price, 'Stop-and-Reverse', entry_time, entry_price, trade_type]
            active_trade = False
        
        # GESTIONE DELLE ENTRATE
        open_new_trade = is_buy_signal or is_sell_signal
        # Apre un nuovo trade se si verifica un segnale e non c'è già un trade attivo (o se è appena stato chiuso per Stop-and-Reverse)
        if open_new_trade and (exit_signal_triggered or not active_trade):
            amount_to_invest = df.loc[current_index, 'Equity'] * risk_per_trade; spread_amount = current_close * spread_pct
            if is_buy_signal: new_trade_type, entry_price_new = 'Long', current_close + spread_amount
            else: new_trade_type, entry_price_new = 'Short', current_close - spread_amount
            if entry_price_new > 0:
                # Calcola la dimensione della posizione in base al rischio per trade definito
                position_size_new = amount_to_invest / entry_price_new
                # Aggiorna lo stato per riflettere il nuovo trade attivo
                active_trade, trade_type, entry_price, position_size = True, new_trade_type, entry_price_new, position_size_new
                entry_time, last_swap_check_date = current_index, current_index.date()
                df.loc[current_index, ['Signal', 'Trade_Type', 'Entry_Price', 'Position_Size', 'Trade_Active']] = [1 if is_buy_signal else -1, new_trade_type, entry_price_new, position_size_new, True]
            
    return df, df['Equity'].iloc[-1]

def _calculate_annualization_factor(df):
    """
    Calcola il fattore di annualizzazione basato sulla frequenza dei dati.
    Ritorna il numero di periodi (candele) in un anno.
    """
    # Se il DataFrame non è valido, non si può calcolare nulla
    if df is None or len(df.index) < 2:
        # Ritorna NaN per indicare che il calcolo non è possibile ed evitare errori
        return np.nan 

    # Calcola la distanza temporale media tra le candele
    time_delta = df.index.to_series().diff().mean()

    if pd.isna(time_delta) or time_delta.total_seconds() == 0:
        return np.nan

    # Calcola il numero di periodi in un anno, rendendo il calcolo valido su qualsiasi timeframe
    return pd.Timedelta(days=365.25) / time_delta

def calculate_metrics(df, initial_capital, df_rf=None):
    """
    Calcola un set di metriche di performance chiave basate sulla curva di equity e sui dati dei trade.
    """
    if df is None or df.empty: return None
    metrics = {}
    
    # Gestisce sia i DataFrame di strategia ('Equity') sia quelli di benchmark ('Equity_BH')
    equity_col = 'Equity' if 'Equity' in df.columns else 'Equity_BH'
    if equity_col not in df.columns:
        print("Errore: Nessuna colonna di equity ('Equity' o 'Equity_BH') trovata.")
        return None
    equity_curve = df[equity_col]

    # Calcoli di performance di base, validi per tutte le strategie
    final_capital = equity_curve.iloc[-1]
    metrics['Total_Return_Percent'] = ((final_capital - initial_capital) / initial_capital) * 100
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    metrics['Max_Drawdown'] = drawdown.min() * 100 if not drawdown.empty else 0
    
    # Calcolo dello Sharpe Ratio annualizzato
    period_returns = equity_curve.pct_change().dropna()
    if not period_returns.empty:
        merged_df = pd.DataFrame({'returns': period_returns})
        if df_rf is not None:
            annualization_factor = _calculate_annualization_factor(df)
            # Converte il tasso risk-free annuale in un tasso periodico corrispondente alla frequenza dei dati (oraria)
            rf_periodic_rate = (1 + df_rf['rate'])**(1 / annualization_factor) - 1
            merged_df = merged_df.join(rf_periodic_rate)
            merged_df['rate'] = merged_df['rate'].fillna(0) # Gestisce eventuali NaN
            excess_returns = merged_df['returns'] - merged_df['rate']
        else:
            excess_returns = period_returns # Senza risk-free, l'excess return è il rendimento stesso
        
        # Calcola lo Sharpe Ratio e lo annualizza
        if excess_returns.std() > 0:
            sharpe_ratio = excess_returns.mean() / excess_returns.std()
            metrics['Sharpe_Ratio_Annualized'] = sharpe_ratio * np.sqrt(_calculate_annualization_factor(df))
        else:
            metrics['Sharpe_Ratio_Annualized'] = 0.0
    else:
        metrics['Sharpe_Ratio_Annualized'] = 0.0

    # Calcola le metriche specifiche dei trade solo se il DataFrame proviene da un backtest di strategia(e non dal Buy & Hold)
    if 'Realized_PnL' in df.columns:
        trades = df[df['Realized_PnL'] != 0].copy()
        metrics['Total_Swap_Cost'] = df['Swap_Cost'].sum()
        metrics['Total_Trades'] = len(trades)
        
        if metrics['Total_Trades'] > 0:
            winning_trades = trades[trades['Realized_PnL'] > 0]
            losing_trades = trades[trades['Realized_PnL'] <= 0]
            metrics['Winning_Trades'] = len(winning_trades)
            metrics['Win_Rate'] = (metrics['Winning_Trades'] / metrics['Total_Trades']) * 100
            total_loss = losing_trades['Realized_PnL'].sum()
            metrics['Profit_Factor'] = abs(winning_trades['Realized_PnL'].sum() / total_loss) if total_loss != 0 else np.inf
        else:
            metrics.update({'Winning_Trades': 0, 'Win_Rate': 0, 'Profit_Factor': 0})
    else:
        # Se non ci sono trade (es. Buy & Hold), imposta le metriche relative a valori neutri o NaN
        metrics['Total_Swap_Cost'] = 0
        metrics['Total_Trades'] = 0
        metrics['Winning_Trades'] = 0
        metrics['Win_Rate'] = float('nan')
        metrics['Profit_Factor'] = float('nan') 

    return metrics

def print_metrics(metrics, name=""):
    """Stampa in modo formattato le metriche di performance."""
    if metrics is None: return
    print("-" * 50); print(f"Metriche: {name}"); print("-" * 50)
    print(f"Rendimento Totale:        {metrics.get('Total_Return_Percent', 0):.2f}%")
    print(f"Costo Totale Swap:        {metrics.get('Total_Swap_Cost', 0):.2f}")
    print(f"Max Drawdown:             {metrics.get('Max_Drawdown', 0):.2f}%")
    print(f"Sharpe Ratio (Ann.):      {metrics.get('Sharpe_Ratio_Annualized', 'N/A')}")
    print(f"Profit Factor:            {metrics.get('Profit_Factor', 0):.2f}")
    print("-" * 25)
    print(f"Trade Totali:             {metrics.get('Total_Trades', 0)}")
    print(f"Win Rate:                 {metrics.get('Win_Rate', 0):.2f}%")
    print("-" * 50)

def print_trades(df):
    """Stampa un elenco dettagliato e formattato di tutti i trade chiusi."""
    trades = df[df['Realized_PnL'] != 0].copy()
    if trades.empty: print("Nessun trade eseguito."); return
    print("\n" + "="*100); print("ELENCO DEI TRADE ESEGUITI"); print("="*100)
    print(f"{'#':<4}{'Entry Time':<22}{'Exit Time':<22}{'Type':<7}{'Exit Type':<18}{'Entry Price':>12}{'Exit Price':>12}{'P&L':>10}")
    print("-"*100)
    for i, (index, trade) in enumerate(trades.iterrows()):
        entry_time_str = trade['Closed_Trade_Entry_Time'].strftime('%Y-%m-%d %H:%M'); exit_time_str = index.strftime('%Y-%m-%d %H:%M')
        print(f"{i+1:<4}{entry_time_str:<22}{exit_time_str:<22}{trade['Closed_Trade_Type']:<7}{trade['Exit_Type']:<18}{trade['Closed_Trade_Entry_Price']:>12.5f}{trade['Exit_Price']:>12.5f}{trade['Realized_PnL']:>10.2f}")
    print("="*100 + "\n")

def calculate_buy_and_hold_equity(df, initial_capital):
    """
    Calcola l'equity di una strategia Buy and Hold sullo stesso DataFrame per confronto.
    """
    if df is None or df.empty:
        return df
        
    df_bh = df.copy()
    
    # Prende il prezzo di chiusura della prima candela come prezzo di acquisto
    start_price = df_bh['Close'].iloc[0]
    
    # Calcola quante "unità" dell'asset si possono acquistare con il capitale iniziale
    number_of_units = initial_capital / start_price
    
    # L'equity in ogni momento è semplicemente il valore corrente delle unità possedute
    df_bh['Equity_BH'] = number_of_units * df_bh['Close']
    
    return df_bh

# Blocco eseguito solo se lo script viene lanciato direttamente (per testare le funzioni)
if __name__ == "__main__":
    print("--- Avvio Backtest Singolo (con confronto Buy and Hold) ---")
    
    # Carica i dati per il test
    df_strategy = fetch_data("DUKA SP500 1H.XLSX", "SP500", "2021-01-01", "2024-12-31")
    
    # Carica i dati del tasso risk-free
    try:
        df_rf = pd.read_excel("DGS3MO.xlsx", header=0, decimal=',')
        df_rf.columns = ['date', 'rate']
        df_rf['date'] = pd.to_datetime(df_rf['date'])
        df_rf.set_index('date', inplace=True)
        df_rf = df_rf[pd.to_numeric(df_rf['rate'], errors='coerce').notna()].copy()
        df_rf['rate'] = pd.to_numeric(df_rf['rate']) / 100
        if not df_rf.empty:
            full_date_range = pd.date_range(start=df_rf.index.min(), end=df_rf.index.max(), freq='D')
            df_rf = df_rf.reindex(full_date_range).ffill()
    except FileNotFoundError:
        print("ATTENZIONE: File DGS3MO.xlsx non trovato. Sharpe Ratio calcolato con R_f = 0.")
        df_rf = None
    except Exception as e:
        print(f"ERRORE durante il caricamento del file risk-free: {e}")
        df_rf = None

    if df_strategy is not None:
        # Imposta i parametri per il test singolo
        commission_params = {'model': 'etoro_per_unit', 'long_fee_per_unit': -1.0, 'short_fee_per_unit': -1.0, 'grace_period_days': 7, 'triple_fee_day': 4}
        
        # Esegue il backtest della strategia
        df_indic = add_indicators(df_strategy, 50, 2.5)
        df_final, _ = generate_signals_with_params(df_indic, spread_pct=0.00015, initial_capital=100000, commission_params=commission_params)
        
        # Calcola la performance del Buy and Hold sullo stesso periodo per confronto
        df_final = calculate_buy_and_hold_equity(df_final, 100000)
        
        # Stampa metriche e trade
        metrics = calculate_metrics(df_final, 100000, df_rf=df_rf)
        print_metrics(metrics, "Test Singolo Strategia")
        print_trades(df_final)
        
        # Genera un grafico di confronto tra la strategia e il benchmark Buy and Hold
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Equity'], mode='lines', name='Strategia Mean Reversion'))
        fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Equity_BH'], mode='lines', name='Buy and Hold (Benchmark)', line=dict(dash='dot', color='grey')))
        
        fig.update_layout(title='Confronto Performance: Strategia vs. Buy and Hold', xaxis_title='Data', yaxis_title='Patrimonio', legend_title='Strategia')
        fig.show()