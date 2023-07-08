# Improve baseline trading strategy using Machine Learning

Reference paper: https://www.sciencedirect.com/science/article/pii/S2405918822000083#appsec2

IV, RV, IV-RV obtained from: https://www.tradingview.com/script/8SovCBDc-rv-iv-vrp/

News API: https://alpaca.markets/docs/market-data/ 

Financial News Sentiment Analysis Model: https://huggingface.co/ProsusAI/finbert

View main.ipynb for feature engineering and data preprocessing 

View Randomforest.ipynb for machine learning

# Insights 

Due to the large amount of content required to be processed by FinBert, sentiment analysis is not used in the random forest machine learning analysis. Research will be continued. 

Current analysis shows that the measure of volatility using features from the paper does not improve the baseline strategy. 