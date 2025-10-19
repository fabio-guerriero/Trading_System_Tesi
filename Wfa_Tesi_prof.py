import pandas as pd
import numpy as np
import plotly.graph_objects as go
from bayes_opt import BayesianOptimization

# Importa le funzioni di base dal modulo di backtesting
from Logica_Codice_Tesi_prof import (
    fetch_data, add_indicators, generate_signals_with_params, 
    calculate_metrics, print_metrics, print_trades, calculate_buy_and_hold_equity
)

def load_and_prepare_rf_data(rf_filepath):
    """
    Carica e prepara i dati del tasso risk-free da un file Excel.
    Gestisce la conversione delle date, la normalizzazione dei tassi e il riempimento dei giorni mancanti.
    """
    try:
        # Carica i dati dal foglio 'Daily' del file Excel
        df_rf = pd.read_excel(
            rf_filepath, 
            header=0,
            decimal=',',
            sheet_name='Daily'
        )
        # Rinomina le colonne per coerenza e facilità d'uso
        df_rf.columns = ['date', 'rate']
        df_rf['date'] = pd.to_datetime(df_rf['date'])
        df_rf.set_index('date', inplace=True)
        
        # Rimuove le righe dove il tasso non è un numero valido (es. '.') e converte la colonna in formato numerico
        df_rf = df_rf[pd.to_numeric(df_rf['rate'], errors='coerce').notna()].copy()
        # Converte il tasso da percentuale a valore decimale (es. 2.5 -> 0.025)
        df_rf['rate'] = pd.to_numeric(df_rf['rate']) / 100
        
        if not df_rf.empty:
            # Crea un range di date completo e riempie i giorni mancanti (es. weekend) con l'ultimo valore valido (forward fill)
            full_date_range = pd.date_range(start=df_rf.index.min(), end=df_rf.index.max(), freq='D')
            df_rf = df_rf.reindex(full_date_range).ffill()
        
        print("Dati del tasso risk-free caricati e preparati correttamente.")
        return df_rf
    except FileNotFoundError:
        # Se il file non viene trovato, procede senza tasso risk-free, avvisando l'utente
        print(f"ATTENZIONE: File risk-free '{rf_filepath}' non trovato. Lo Sharpe Ratio sarà calcolato con R_f = 0.")
        return None
    except Exception as e:
        # Gestisce altri errori imprevisti durante il caricamento
        print(f"ERRORE durante il caricamento del file risk-free: {e}")
        return None

def run_bayesian_optimization_on_period(data, param_bounds, optimization_metric, fixed_params, init_points, n_iter, verbose_level=2):
    """
    Esegue l'ottimizzazione bayesiana dei parametri della strategia su un dato periodo di tempo (In-Sample).
    Cerca la combinazione di parametri che massimizza una metrica di performance specificata.
    """
    def objective_function(moving_average_period, std_dev_multiplier):
        # Funzione obiettivo che l'ottimizzatore bayesiano cercherà di massimizzare.
        # Esegue un backtest completo per ogni combinazione di parametri testata.
        
        # Definisce i parametri correnti
        current_strategy_params = {'moving_average_period': int(round(moving_average_period)), 'std_dev_multiplier': std_dev_multiplier}
        
        # Esegue il backtest con i parametri correnti
        df_with_indicators = add_indicators(data.copy(), **current_strategy_params)
        df_with_signals, _ = generate_signals_with_params(
            df_with_indicators, spread_pct=0.0, initial_capital=fixed_params['initial_capital'],
            risk_per_trade=fixed_params['risk_per_trade'], commission_params={'model': 'none'}
        )
        metrics = calculate_metrics(df_with_signals, fixed_params['initial_capital'], df_rf=None)
        
        # Gestione dei casi anomali: si assegnano penalità (tramite valori molto negativi) per escludere set di parametri non validi
        if metrics is None: return -99999.0  # Penalità se il backtest fallisce
        if metrics.get('Total_Trades', 0) < fixed_params['min_trades']: return -88888.0  # Penalità se non vengono eseguiti abbastanza trade
        if metrics.get('Max_Drawdown', 0) < -abs(fixed_params['max_drawdown_pct']): return -77777.0  # Penalità se il drawdown supera la soglia massima
        
        metric_value = metrics.get(optimization_metric, -99999.0)
        
        # Penalità se la metrica non è un numero finito (es. NaN, Inf)
        if not np.isfinite(metric_value): return -66666.0
        
        return metric_value
    
    # Inizializza l'ottimizzatore. La verbosità (cioè quanti log stampare) è controllata dal parametro 'verbose_level'
    optimizer = BayesianOptimization(f=objective_function, pbounds=param_bounds, random_state=1, verbose=verbose_level)
    
    if verbose_level > 0:
        print(f"  > Inizio ottimizzazione bayesiana ({init_points} init, {n_iter} iter)...")
    
    # Avvia il processo di massimizzazione
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    # Se l'ottimizzazione non ha trovato alcun massimo valido o il risultato è negativo, ritorna None
    if optimizer.max is None or not optimizer.max or optimizer.max['target'] < 0:
        return None, -np.inf
        
    best_params = optimizer.max['params']
    # Assicura che il periodo della media mobile sia un intero anche nel risultato finale
    best_params['moving_average_period'] = int(round(best_params['moving_average_period']))
    
    return best_params, optimizer.max['target']

def run_walk_forward_analysis(
    full_df, df_rf, in_sample_period_days, out_of_sample_period_days, 
    param_bounds, optimization_metric, fixed_params, init_points, n_iter, show_opt_log=True
):
    """
    Orchestra l'analisi Walk-Forward, dividendo i dati in finestre In-Sample (per l'ottimizzazione) 
    e Out-of-Sample (per il test), facendo scorrere queste finestre nel tempo.
    """
    all_oos_results = []
    
    # L'analisi inizia dalla prima data disponibile nel set di dati fornito
    start_date = full_df.index.min() 
    # L'analisi termina all'ultima data disponibile
    end_date = full_df.index.max() 
    
    in_sample_offset = pd.DateOffset(days=in_sample_period_days)
    out_of_sample_offset = pd.DateOffset(days=out_of_sample_period_days)
    
    current_date = start_date
    # Il capitale viene aggiornato dinamicamente finestra dopo finestra, partendo dal capitale iniziale
    wfa_capital = fixed_params['initial_capital']
    # Mantiene in memoria l'ultimo set di parametri validi in caso un'ottimizzazione fallisca
    window_num, last_successful_params = 1, None
    # Imposta il livello di verbosità per l'ottimizzatore in base alla configurazione
    verbose_level = 2 if show_opt_log else 0

    # Il ciclo continua finché c'è abbastanza spazio per una finestra In-Sample completa prima della data di fine
    while current_date + in_sample_offset < end_date:
        # Definizione delle finestre temporali per il ciclo corrente
        in_sample_start = current_date
        in_sample_end = current_date + in_sample_offset
        oos_start = in_sample_end
        # Assicura che la finestra Out-of-Sample non superi la data finale del backtest
        oos_end = min(oos_start + out_of_sample_offset, end_date)

        print("-" * 80); print(f"FINESTRA WFA #{window_num} | IS: {in_sample_start.date()}->{in_sample_end.date()} | OOS: {oos_start.date()}->{oos_end.date()}"); print("-" * 80)
        
        # Seleziona i dati per il periodo In-Sample (ottimizzazione) e Out-of-Sample (test)
        df_in_sample = full_df.loc[in_sample_start:in_sample_end].copy()
        df_oos = full_df.loc[oos_start:oos_end].copy()
        
        # Salta la finestra se uno dei due periodi non contiene dati
        if df_in_sample.empty or df_oos.empty:
            print("  ! Finestra saltata: periodo vuoto."); current_date += out_of_sample_offset; window_num += 1; continue

        # Esegue l'ottimizzazione sul periodo In-Sample
        best_params, _ = run_bayesian_optimization_on_period(df_in_sample, param_bounds, optimization_metric, fixed_params, init_points, n_iter, verbose_level=verbose_level)
        
        # Utilizza i nuovi parametri ottimali se trovati, altrimenti riutilizza quelli dell'ultima finestra valida
        current_params_to_use = best_params if best_params is not None else last_successful_params
        
        # Se non ci sono parametri disponibili (né nuovi né vecchi), salta la finestra per evitare errori
        if current_params_to_use is None:
            print("  ! ERRORE CRITICO: Nessun parametro valido. Salto finestra."); current_date += out_of_sample_offset; window_num += 1; continue
        
        if best_params is not None:
            print(f"  > Nuovi parametri ottimali trovati: {current_params_to_use}")
            last_successful_params = current_params_to_use.copy()
        else:
            print(f"  ! Nessun nuovo parametro. RIUTILIZZO dei precedenti: {last_successful_params}")

        print("  > Esecuzione test Out-of-Sample...")
        oos_strategy_params = {'moving_average_period': current_params_to_use['moving_average_period'], 'std_dev_multiplier': current_params_to_use['std_dev_multiplier']}
        
        # Applica i parametri trovati al periodo Out-of-Sample
        df_oos_with_indicators = add_indicators(df_oos, **oos_strategy_params)
        df_oos_result, final_capital_oos = generate_signals_with_params(
            df_oos_with_indicators, spread_pct=fixed_params['spread_pct_out_of_sample'], initial_capital=wfa_capital,
            risk_per_trade=fixed_params['risk_per_trade'], commission_params=fixed_params['commission_params_out_of_sample']
        )
        
        # Aggiorna il capitale per la finestra successiva con il risultato del test OOS corrente
        wfa_capital = final_capital_oos
        all_oos_results.append(df_oos_result)
        print(f"  > Fine test OOS. Capitale finale: {wfa_capital:,.2f}")
        
        # Fa scorrere la finestra temporale in avanti della durata del periodo Out-of-Sample
        current_date += out_of_sample_offset
        window_num += 1
    
    if not all_oos_results: print("\nNessun risultato OOS generato."); return None
    
    # Concatena i risultati di tutte le finestre Out-of-Sample in un unico DataFrame
    return pd.concat(all_oos_results)

if __name__ == "__main__":
    # PANNELLO DI CONTROLLO
    # Flag per attivare/disattivare le diverse sezioni dell'analisi
    RUN_WFA = True
    RUN_STATIC_BACKTEST = True
    RUN_BUY_AND_HOLD = True
    
    # Flag per controllare la verbosità e la stampa dei dettagli dei trade
    SHOW_OPTIMIZATION_LOG = False
    SHOW_WFA_TRADES = True
    SHOW_STATIC_TRADES = True
    
    print("--- Avvio Analisi Flessibile ---")
    
    # CONFIGURAZIONE GENERALE
    
    # Data di inizio del periodo di test effettivo (Out-of-Sample). I dati precedenti verranno usati per la prima ottimizzazione.
    OOS_START_DATE = "2014-01-01" 
    END_DATE = "2017-12-31"
    
    # Durata delle finestre di ottimizzazione (In-Sample) e di test (Out-of-Sample)
    IN_SAMPLE_DAYS = 120
    OUT_OF_SAMPLE_DAYS = 90
    
    # Carica l'intero storico dei dati dal file Excel
    df_full_data = fetch_data("DUKA SP500 1H.XLSX", "SP500")
    df_rf = load_and_prepare_rf_data("DGS3MO.xlsx")

    # VALIDAZIONE E PREPARAZIONE DEI DATI
    if df_full_data is not None:
        # Calcola la data da cui sono necessari i dati per poter eseguire la prima ottimizzazione In-Sample
        oos_start_dt = pd.to_datetime(OOS_START_DATE)
        required_data_start_dt = oos_start_dt - pd.DateOffset(days=IN_SAMPLE_DAYS)
        
        # Controlla se lo storico disponibile è sufficiente per iniziare l'analisi
        if required_data_start_dt < df_full_data.index.min():
            print(f"ERRORE: La data di inizio del test ({OOS_START_DATE}) richiede dati a partire dal {required_data_start_dt.date()},")
            print(f"ma i dati disponibili iniziano solo il {df_full_data.index.min().date()}.")
            exit() # Interrompe l'esecuzione se i dati non sono sufficienti

        # Filtra il DataFrame per includere solo il periodo necessario per l'intera WFA (dati per la prima ottimizzazione + dati di test)
        df_full = df_full_data.loc[required_data_start_dt:END_DATE].copy()
        
        # Crea un DataFrame separato per i benchmark (B&H, Statico) che parte esattamente dalla data di inizio del test OOS
        df_benchmarks = df_full_data.loc[OOS_START_DATE:END_DATE].copy()

        # Dizionario dei parametri fissi, non soggetti a ottimizzazione
        fixed_params = {
            'initial_capital': 10000, 'risk_per_trade': 1, 'min_trades': 3,
            'max_drawdown_pct': 20.0, 'spread_pct_out_of_sample': 0.00015,
            'commission_params_out_of_sample': {'model': 'etoro_per_unit', 'long_fee_per_unit': -1, 'short_fee_per_unit': -1, 'grace_period_days': 7, 'triple_fee_day': 4}
        }
        # Range dei parametri che verranno ottimizzati 
        param_bounds_to_optimize = {'moving_average_period': (20, 70), 'std_dev_multiplier': (1.0, 3.0)}
        INIT_POINTS, N_ITERATIONS = 25, 75
        OPTIMIZATION_METRIC = 'Sharpe_Ratio_Annualized'
        # Parametri fissi usati per il backtest statico di confronto
        static_strategy_params = { 'moving_average_period': 50, 'std_dev_multiplier': 2 }
        
        # ESECUZIONE E RACCOLTA RISULTATI
        final_results = []
        wfa_results_df, df_static_results, df_bh_results = None, None, None
        
        if RUN_BUY_AND_HOLD:
            print("\n--- 1. Calcolo del benchmark Buy and Hold ---")
            # Esegue il calcolo del Buy & Hold sul DataFrame dei benchmark
            df_bh_results = calculate_buy_and_hold_equity(df_benchmarks, fixed_params['initial_capital'])
            bh_metrics = calculate_metrics(df_bh_results, fixed_params['initial_capital'], df_rf)
            final_results.append({'name': "Benchmark Buy and Hold", 'metrics': bh_metrics})

        if RUN_STATIC_BACKTEST:
            print("\n--- 2. Esecuzione del backtest con parametri statici ---")
            # Esegue il backtest statico sul DataFrame dei benchmark
            df_static_indicators = add_indicators(df_benchmarks.copy(), **static_strategy_params)
            df_static_results, _ = generate_signals_with_params(
                df_static_indicators, spread_pct=fixed_params['spread_pct_out_of_sample'],
                initial_capital=fixed_params['initial_capital'], commission_params=fixed_params['commission_params_out_of_sample']
            )
            static_metrics = calculate_metrics(df_static_results, fixed_params['initial_capital'], df_rf)
            final_results.append({'name': f"Strategia Statica (Parametri: {static_strategy_params})", 'metrics': static_metrics})

        if RUN_WFA:
            print("\n--- 3. Esecuzione della Walk-Forward Analysis ---")
            # Esegue la WFA, che richiede il set di dati completo (df_full) per avere dati sufficienti per la prima finestra IS
            wfa_results_df = run_walk_forward_analysis(
                full_df=df_full, df_rf=df_rf, in_sample_period_days=IN_SAMPLE_DAYS, out_of_sample_period_days=OUT_OF_SAMPLE_DAYS,
                param_bounds=param_bounds_to_optimize, optimization_metric=OPTIMIZATION_METRIC,
                fixed_params=fixed_params, init_points=INIT_POINTS, n_iter=N_ITERATIONS,
                show_opt_log=SHOW_OPTIMIZATION_LOG
            )
            if wfa_results_df is not None:
                wfa_metrics = calculate_metrics(wfa_results_df, fixed_params['initial_capital'], df_rf=df_rf)
                final_results.append({'name': f"Strategia WFA su SP500", 'metrics': wfa_metrics})
                
        # STAMPA RIEPILOGO FINALE E PLOT 
        print("\n" + "="*50); print("RIEPILOGO COMPARATIVO FINALE"); print("="*50)
        for result in final_results:
            print_metrics(result['metrics'], result['name'])
        
        if SHOW_STATIC_TRADES and df_static_results is not None:
            print("\n" + "="*80); print("DETTAGLIO TRADE DELLA STRATEGIA STATICA"); print("="*80)
            print_trades(df_static_results)
        if SHOW_WFA_TRADES and wfa_results_df is not None:
             print("\n" + "="*80); print("DETTAGLIO TRADE DELLA WALK-FORWARD ANALYSIS"); print("="*80)
             print_trades(wfa_results_df)

        if RUN_WFA or RUN_STATIC_BACKTEST or RUN_BUY_AND_HOLD:
            print("\n--- Generazione Grafico Comparativo ---")
            fig = go.Figure()

            # Aggiunge le curve di equity al grafico solo se le relative analisi sono state eseguite
            if RUN_WFA and wfa_results_df is not None:
                fig.add_trace(go.Scatter(x=wfa_results_df.index, y=wfa_results_df['Equity'], mode='lines', name='Strategia WFA (Adattiva)', line=dict(color='blue', width=2)))
            if RUN_STATIC_BACKTEST and df_static_results is not None:
                fig.add_trace(go.Scatter(x=df_static_results.index, y=df_static_results['Equity'], mode='lines', name='Strategia Statica', line=dict(color='orange', dash='dash')))
            if RUN_BUY_AND_HOLD and df_bh_results is not None:
                fig.add_trace(go.Scatter(x=df_bh_results.index, y=df_bh_results['Equity_BH'], mode='lines', name='Buy and Hold (Passiva)', line=dict(color='grey', dash='dot')))

            fig.update_layout(title=f'Confronto Performance Strategie - SP500', xaxis_title='Data', yaxis_title='Capitale', legend_title="Tipo di Strategia")
            fig.show()