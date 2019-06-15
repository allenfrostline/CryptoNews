import numpy as np
import pandas as pd
from collections import Counter
from scipy.optimize import minimize
from wordcloud import STOPWORDS


class BacktestEngine:
    def __init__(self):
        df = pd.read_csv('data/posts.csv', index_col=0)
        df.index = pd.to_datetime(df.index).date
        df_eod = df.copy()
        df_eod['date'] = pd.to_datetime(df_eod.index).date
        g = df_eod.groupby('date')
        title = g.title.apply(' '.join)
        abstract = g.abstract.apply(' '.join)
        content = g.content.apply(lambda x: '\n\n'.join(map(str, x.values)))
        df_eod = pd.concat([title, abstract, content], axis=1)
        df_eod.index = pd.to_datetime(df_eod.index)
        df_eod.index.name = None
        df_eod.title = df_eod.title.str.lower()
        df_eod.abstract = df_eod.abstract.str.lower()
        df_eod.content = df_eod.content.str.lower()
        remove_abbr = lambda x: ' '.join([_.split('’')[0] for _ in x.split()])
        df_eod = df_eod.applymap(remove_abbr)
        alph_space = lambda x: (ord('a') <= ord(x) <= ord('z')) or x == ' '
        keep_alph = lambda x: ''.join(filter(alph_space, x))
        df_eod = df_eod.applymap(keep_alph)
        replace_stop_words = lambda x: ' '.join(_ for _ in x.split() if _ not in STOPWORDS)
        df_eod = df_eod.applymap(replace_stop_words)
        df_eod = df_eod.applymap(str.split).applymap(sorted)
        price = pd.read_csv('data/price.csv', index_col=0)
        price.index = pd.to_datetime(price.index)
        mkcap = pd.read_csv('data/mkcap.csv', index_col=0)
        mkcap.index = pd.to_datetime(mkcap.index)
        self.data = (price, mkcap, df_eod)  # store data for later usage

    def run(self, lookback_window=60, minimum_occurrence=50, minimum_length=5, alpha=.1, size=lambda x: 2 * x - 1):
        n = 3  # number of coins to hold (long/short)
        d = lookback_window  # number of days to train
        price, mkcap, df_eod = self.data  # data stored for training (reusable)

        rtn = price.pct_change().fillna(0)
        rtn.index.name = None
        temp = (rtn * (mkcap.agg('argsort') < 10)).rolling(d).sum().T
        coins = pd.DataFrame(
            [temp.nlargest(n, t).index.tolist() + temp.nsmallest(n, t).index.tolist() for t in rtn.index],
            index=rtn.index).shift().dropna()  # shift to avoid forward-looking bias

        coef = np.array([1] * n + [-1] * n) / n
        rtn = pd.DataFrame([rtn.loc[t, coins.loc[t].values] @ coef for t in coins.index], index=coins.index)[0]
        rtn.name = 'r'

        df_total = df_eod[['content']].join(rtn, how='inner').dropna()
        df_total['p'] = np.argsort(df_total.iloc[:, 1]) / (df_total.shape[0] - 1)
        df_total.columns = ['words', 'r', 'p']

        test_split = 0.3
        len_total = len(df_total)
        len_test = round(len_total * test_split)
        len_train = len_total - len_test

        df_train = df_total.iloc[:len_train, :]
        df_test = df_total.iloc[-len_test:, :]

        words_all = sorted(list(set(w for words in df_train.words for w in words)))
        words_dict = {w: [] for w in words_all}

        for i in range(df_train.shape[0]):
            words, r, _ = df_train.iloc[i]
            for w in words: words_dict[w].append(r > 0)

        words_df = pd.DataFrame({
            'word': words_all,
            'r': [np.mean(words_dict[w]) for w in words_all],
            'k': [len(words_dict[w]) for w in words_all]
        }, index=range(len(words_all))).sort_values(by='r')

        words_df = words_df[words_df.k >= minimum_occurrence][words_df.word.apply(len) >= minimum_length].reset_index(drop=True)
        words_df['p'] = np.argsort(words_df.r) / (len(words_df) - 1)

        pos_words = words_df[words_df.p > 1 - alpha].word.values.tolist()
        neg_words = words_df[words_df.p < alpha].word.values.tolist()

        keywords = pos_words + neg_words

        def count_words(words):
            c = Counter(words)
            d = [c[w] for w in keywords]
            return np.array(d)

        df_train['d'] = df_train.words.apply(count_words).values

        p = df_train.p.values
        W = np.vstack([p, 1 - p]).T
        D = np.vstack(df_train.d.values)
        D = (D.T / D.sum(axis=1)).T
        D[np.isnan(D)] = 0
        O_pos, O_neg = np.maximum(np.linalg.pinv(W.T @ W) @ (W.T @ D), 0)
        O_pos = (O_pos / O_pos.sum())
        O_neg = (O_neg / O_neg.sum())

        def pen_ll(p, d, lamda):
            return (d * np.log(p * O_pos + (1 - p) * O_neg)).sum() + lamda * np.log(p * (1 - p))

        def predict(d, lamda):
            return minimize(lambda p: -pen_ll(p, d, lamda), x0=.5, method='Nelder-Mead').x[0]

        def mse_p(lamda):
            p_hat = np.array([predict(d, lamda) for d in df_train.d])
            return ((df_train.p.values - p_hat)**2).mean()

        best_lamda = minimize(mse_p, x0=0, method='Nelder-Mead', bounds=(0, None)).x[0]

        df_test['d'] = df_test.words.apply(count_words).values
        df_test['p_hat'] = [predict(d, best_lamda) for d in df_test.d]
        df_test = df_test[['words', 'r', 'p', 'p_hat', 'd']]

        compound = lambda x: (1 + x).cumprod() - 1

        r = df_test.r
        size = df_test.p_hat.apply(size)

        daily_rtn = r * size
        cum_rtn = compound(daily_rtn)
        daily_rtn_momentum = rtn[cum_rtn.index]
        cum_rtn_momentum = compound(daily_rtn_momentum)
        daily_rtn_momentum_rev = -daily_rtn_momentum
        cum_rtn_momentum_rev = compound(daily_rtn_momentum_rev)
        daily_rtn_buy_hold = price.BTC.pct_change().loc[cum_rtn.index]
        cum_rtn_buy_hold = compound(daily_rtn_buy_hold)

        win_rate = lambda x: (x > 0).mean()
        sharpe = lambda x: x.mean() / x.std() * np.sqrt(360)
        mdd = lambda x: (1 - x / x.cummax()).max()

        DAILY = [daily_rtn_buy_hold, daily_rtn_momentum, daily_rtn_momentum_rev, daily_rtn]
        CUM = [cum_rtn_buy_hold, cum_rtn_momentum, cum_rtn_momentum_rev, cum_rtn]
        wr = [win_rate(x) for x in DAILY]
        sr = [sharpe(x) for x in DAILY]
        md = [mdd(x + 1) for x in CUM]
        result = pd.DataFrame({'Win%': wr, 'Sharpe': sr, 'MDD': md}, index=['Buy & Hold', 'Momentum', 'Mom. Rev.', 'Strategy'])
        result.index.name = 'Statistics'

        return result