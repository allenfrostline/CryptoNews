import os
import json
import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
from _datetime import datetime, date, timedelta


def save_crypto_coins_history(i_rank_start=1, i_rank_end=10, i_coin_file_path='crypto_coins', i_from_date=None, i_to_date=None, i_min_volume=100000, i_coin_markets=[]):
    from_date, to_date = get_from_to_dates(i_from_date, i_to_date)
    rank_range_from_start = i_rank_end - i_rank_start + 1
    coins_ranking_dict = get_coins_current_ranking(i_rank_start, rank_range_from_start, i_min_volume)
    df_coins = pd.DataFrame([])
    for rank, coin in coins_ranking_dict.items():
        if is_coin_in_markets(coin, set(i_coin_markets)):
            df_coins = df_coins.append(get_coins_historical_data(rank, coin, from_date, to_date))
            write_df_to_csv(df_coins, i_coin_file_path + '.csv')


def get_coins_current_ranking(i_start, i_limit, i_min_volume):
    url_coin_list_json = 'https://api.coinmarketcap.com/v1/ticker/?start={}&limit={}'.format(i_start - 1, i_limit)
    page = requests.get(url_coin_list_json)
    json_file = json.loads(page.text)
    coins_dict = {}
    for k in json_file:
        if float(k['24h_volume_usd']) >= i_min_volume:
            coins_dict[k['rank']] = k['id']
    return coins_dict


def get_coins_historical_data(i_rank, i_coin, i_from_date, i_to_date):
    df_coin = get_specific_coin_historical_data(i_coin, i_from_date, i_to_date)
    df_coin['Coin'] = i_coin
    df_coin['Cur. Rank'] = i_rank
    df_coin = pd.concat([df_coin.iloc[:, 7:], df_coin.iloc[:, 0:7]], axis=1, join_axes=[df_coin.index])
    return df_coin


def get_specific_coin_historical_data(i_coin, i_from_date, i_to_date):
    currencies = "https://coinmarketcap.com/currencies/"
    currencies_end = '/historical-data/'
    dates = '?start={}&end={}'.format(i_from_date, i_to_date)
    url = currencies + i_coin + currencies_end + dates
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    table = soup.find('table')
    data = {'Date': [], 'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': [], 'Market Cap': []}
    try:
        rows = table.findAll('tr')[1:]
        for row in rows:
            cols = row.findAll('td')
            data['Date'].append(cols[0].string)
            data['Open'].append(cols[1].string)
            data['High'].append(cols[2].string)
            data['Low'].append(cols[3].string)
            data['Close'].append(cols[4].string)
            data['Volume'].append(cols[5].string)
            data['Market Cap'].append(cols[6].string)
        coin_data = pd.DataFrame(data)
    except AttributeError:
        print('input parameters not valid')
        sys.exit(13)
    return coin_data


def write_df_to_csv(i_df, i_file):
    try:
        i_df.to_csv(i_file)
    except IOError as e:
        print(e)
        sys.exit(13)


def get_from_to_dates(i_from_date, i_to_date):
    try:
        if i_from_date is None:
            from_date = str(date.today() + timedelta(days=-30))
        else:
            from_date = i_from_date
        from_date = datetime.strptime(from_date, '%Y-%m-%d').strftime('%Y%m%d')

        if i_to_date is None:
            to_date = str(date.today() + timedelta(days=-1))
        else:
            to_date = i_to_date
        to_date = datetime.strptime(to_date, '%Y-%m-%d').strftime('%Y%m%d')

        return from_date, to_date
    except ValueError as e:
        print(e)
        sys.exit(13)


def is_coin_in_markets(i_coin, i_coin_markets_to_search):
    coin_in_markets = False
    coin_markets_url = 'https://coinmarketcap.com/currencies/{}/#markets'.format(i_coin)
    if not i_coin_markets_to_search:
        coin_in_markets = True
    else:
        page = requests.get(coin_markets_url)
        soup = BeautifulSoup(page.text, 'html.parser')
        table = soup.find('table')
        rows = table.findAll('tr')[1:]
        markets = set()
        for row in rows:
            cols = row.findAll('td')
            if cols[1].text is not None:
                markets.add(cols[1].text.upper())

            for market in i_coin_markets_to_search:
                if market.upper() in markets:
                    coin_in_markets = True
                    break

    return coin_in_markets


if __name__ == '__main__':
    symbols = [
        ('BTC', 'bitcoin'),
        ('DOGE', 'dogecoin'),
        ('FTC', 'feathercoin'),
        ('IXC', 'ixcoin'),
        ('LTC', 'litecoin'),
        ('MEC', 'megacoin'),
        ('NMC', 'namecoin'),
        ('NVC', 'novacoin'),
        ('NXT', 'nxt'),
        ('OMNI', 'omni'),
        ('PPC', 'peercoin'),
        ('XPM', 'primecoin'),
        ('XRP', 'ripple'),
    ]
    price = dict()
    mkcap = dict()
    for sym, i_coin in symbols:
        if not os.path.isfile(f'{i_coin}.csv'):
            try:
                print(f'Downloading {i_coin:>12}', end=': ')
                df = get_specific_coin_historical_data(i_coin, '20120101', '20190520')[::-1]
                print(df.Date.iloc[0], df.Date.iloc[-1])
                df.to_csv(f'{i_coin}.csv')
            except:
                print('failed')
                pass

    for sym, i_coin in symbols:
        print(f'Loading {i_coin:>12}', end=': ')
        df = pd.read_csv(f'{i_coin}.csv').set_index('Date')
        df.index = pd.to_datetime(df.index)
        print(df.index[0].date(), df.index[-1].date())
        price[sym] = df['Close']
        mkcap[sym] = df['Market Cap']

    def convert(x):
        s = str(x).replace(',', '')
        try:
            return float(s)
        except:
            if s == '-':
                return float('nan')

    price = pd.DataFrame(price).dropna().applymap(convert).fillna(method='ffill')
    price.index.name = None
    price.to_csv('../data/price.csv')

    mkcap = pd.DataFrame(mkcap).dropna().applymap(convert).fillna(method='ffill')
    mkcap.index.name = None
    mkcap.to_csv('../data/mkcap.csv')

    print(f'Task finished: {len(symbols)} coins in total')
