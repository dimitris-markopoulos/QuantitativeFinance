from ml_pipeline import SVMTradingPipeline

def run_full_backtest(ticker='NKE',
                      start='2010-01-01',
                      end='2025-04-22',
                      test_prop=0.05,
                      initial_capital=10000,
                      plot_path='media/model_vs_benchmark_PnL.png',
                      runtime='fast'):

    pipeline = SVMTradingPipeline(ticker, start_date=start, end_date=end, test_prop=test_prop)
    
    # Call pipeline methods
    pipeline.download_data
    pipeline.engineer_features
    pipeline.split_data

    # Cross-validate and fit
    best_params = pipeline.cross_validate(svm_param_grid=None, n_splits=5, runtime=runtime)
    pipeline.fit_predict(param_grid=best_params, plot_distribution=False, save_path=None)

    # Simulate and plot
    pnl_df = pipeline.simulate_trading(initial_capital=initial_capital, benchmark=True)
    pipeline.plot_pnl(pnl_df, save_path=plot_path)

    return pipeline, pnl_df