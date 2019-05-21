# Hybrid Sentiment-Momentum Strategy in Cryptocurrency Market

This is an exploratory research trying to implement a simple yet robust sentiment analysis mechanism in the cryptocurrency market. We call it a hybrid sentiment-momentum strategy as it applies context analysis (or more specifically, word analysis) into traditional momentum strategies and adjusts position automatically over time using both factors. The hybrid strategy yields a remarkable Sharpe ratio of 2.68 with acceptable maximum drawdown in out-of-sample backtest, which is a huge improvement from classical momentum strategies in our analysis.

<p align='center' ><img src="/misc/wordcloud.png" width=90%/></p>
<p align='center' ><b>Fig. 1:</b> Cloud of Positive/Negative Words</p>
<p align='center' ><img src="/misc/performance.png" width=100%/></p>
<p align='center' ><b>Fig. 2:</b> Backtest Performance (also: see table below)</p>

| Statistics | Buy & Hold | Momentum  | Mom. Rev. | Strategy |  
| :-:        | :-:        | :-:       | :-:       | :-:      |  
| Win %      | 0.524051   | 0.437975  | 0.562025  | 0.536709 |  
| Sharpe     | -0.975205  | -0.605331 | 0.605331  | 2.677082 |  
| MDD        | 0.826419   | 0.970956  | 0.947963  | 0.206906 |  
