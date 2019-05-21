# Hybrid Sentiment-Momentum Strategy in Cryptocurrency Market

This is an exploratory research trying to implement a simple yet robust sentiment analysis mechanism in the cryptocurrency market. We call it a hybrid sentiment-momentum strategy as it applies context analysis (or more specifically, word analysis) into traditional momentum strategies and adjusts position automatically over time using both factors. The hybrid strategy yields a remarkable Sharpe ratio of 2.68 with acceptable maximum drawdown in out-of-sample backtest, which is a huge improvement from classical momentum strategies in our analysis.

<p align='center' ><img src="/misc/wordcloud.png" width=90%/></p>
<p align='center' ><b>Fig. 1:</b> Cloud of Positive/Negative Words</p>
<p align='center' ><img src="/misc/performance.png" width=100%/></p>
<p align='center' ><b>Fig. 2:</b> Backtest Performance (also: see table below)</p>

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border:none;border-color:#ccc;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-buh4{background-color:#f9f9f9;text-align:left;vertical-align:top}
.tg .tg-fymr{font-weight:bold;border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table style="border-collapse:collapse;border-spacing:0;border:none;border-color:#ccc;">
  <tr>
    <td style="font-weight:bold;border-color:inherit;text-align:left;vertical-align:top">Statistics</td>
    <td style="font-weight:bold;border-color:inherit;text-align:left;vertical-align:top">Buy &amp; Hold</td>
    <td style="font-weight:bold;border-color:inherit;text-align:left;vertical-align:top">Momentum</td>
    <td style="font-weight:bold;border-color:inherit;text-align:left;vertical-align:top">Mom. Rev.</td>
    <td style="font-weight:bold;border-color:inherit;text-align:left;vertical-align:top">Strategy</td>
  </tr>
  <tr>
    <td style="background-color:#f9f9f9;text-align:left;vertical-align:top">Win %</td>
    <td style="background-color:#f9f9f9;text-align:left;vertical-align:top">0.524051</td>
    <td style="background-color:#f9f9f9;text-align:left;vertical-align:top">0.437975</td>
    <td style="background-color:#f9f9f9;text-align:left;vertical-align:top">0.562025</td>
    <td style="background-color:#f9f9f9;text-align:left;vertical-align:top">0.536709</td>
  </tr>
  <tr>
    <td style="text-align:left;vertical-align:top">Sharpe</td>
    <td style="text-align:left;vertical-align:top">-0.975205</td>
    <td style="text-align:left;vertical-align:top">-0.605331</td>
    <td style="text-align:left;vertical-align:top">0.605331</td>
    <td style="text-align:left;vertical-align:top">2.677082</td>
  </tr>
  <tr>
    <td style="background-color:#f9f9f9;text-align:left;vertical-align:top">MDD</td>
    <td style="background-color:#f9f9f9;text-align:left;vertical-align:top">0.826419</td>
    <td style="background-color:#f9f9f9;text-align:left;vertical-align:top">0.970956</td>
    <td style="background-color:#f9f9f9;text-align:left;vertical-align:top">0.947963</td>
    <td style="background-color:#f9f9f9;text-align:left;vertical-align:top">0.206906</td>
  </tr>
</table>