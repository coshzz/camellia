from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import zipfile
from pathlib import Path
import argparse


class Historical_spot_data:
    
    def __init__(self, local_path="."):
        self.__local_path = local_path
        
        
    def read_klines_data(self, csv_file):
        cols_name = [
            "open_time", "open", "high", "low", "close", "volume", 
            "close_time", "quote_asset_volume", "number_of_trades", 
            "taker_buy_base_asset_volume", 
            "taker_buy_quote_asset_volume", 
            "ignore"]
        df = pd.read_csv(csv_file, names=cols_name)
        for col in ['open_time', 'close_time']:
            df[col] = pd.to_datetime(df[col], unit='ms')
        df1 = df.set_index("open_time")[["open","high","low","close","volume"]].copy()
        
        return df1

    """
    Download klines spot (zip file) from Binance and return Bytes content.
    
    spot:
    https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2022-06-13.zip
    
    Ex.
    pair = ['BTCUSDT', 'ETHUSDT']
    interval = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    date = '2022-01-01'
    """
    def dl_klines_spot_daily_from_binance(self, pair: str = 'BTCUSDT', 
                                interval: str = '1m', 
                                date: str = '2022-01-01'):
        
        base_url = "https://data.binance.vision"
        url_tmplt = base_url + "/data/spot/daily/klines/{pair}/{interval}/{pair}-{interval}-{date}.zip"
        
        url = url_tmplt.format(interval=interval, pair=pair, date=date)
        
        print(f"downloading: {url}")
        resp = requests.get(url)
        
        if resp.status_code==200:
            print("download complete")
            return resp.content
        else:
            print(f"download error {resp.status_code}")
            return None
            
            
    """
    Download klines spot (zip file) from Binance and return Bytes content.
    
    spot:
    https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2022-06.zip
    
    Ex.
    pair = ['BTCUSDT', 'ETHUSDT']
    interval = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    month = '2022-01'
    """
    def dl_klines_spot_monthly_from_binance(self, pair: str = 'BTCUSDT', 
                                interval: str = '1m', 
                                month: str = '2022-01'):
        
        if len(month)>7:
            print(f"warning: len(month)>7 = '{month}'")
            month = month[:7]
        
        base_url = "https://data.binance.vision"
        url_tmplt = base_url + "/data/spot/monthly/klines/{pair}/{interval}/{pair}-{interval}-{month}.zip"
        
        url = url_tmplt.format(interval=interval, pair=pair, month=month)
        
        print(f"downloading: {url}")
        resp = requests.get(url)
        
        if resp.status_code==200:
            print("download complete")
            return resp.content
        else:
            print(f"download error {resp.status_code}")
            return None
        
    def list_daily_interval(self, from_date: str = '2022-01-01', 
                                    to_date: str = '2022-01-31'):
        # convert to datetime
        from_date = datetime.strptime(from_date, '%Y-%m-%d')
        to_date = datetime.strptime(to_date, '%Y-%m-%d')
        
        # limit 'to_date' to now()
        if to_date>datetime.now():
            to_date = datetime.now()
        
        # check order date
        if from_date<=to_date:
            days = (to_date - from_date).days
        else:
            print("Please check the 'from_date' after 'to_date'")
            days = (from_date - to_date).days
    
        # dataframe container
        list_days = []
        
        # loop over days
        for day in range(days):
            current_date = from_date + relativedelta(days=day)
            list_days.append(datetime.strftime(current_date, "%Y-%m-%d"))
        
        return list_days
            
    def list_monthly_interval(self, from_month: str = '2022-01', 
                                    to_month: str = '2022-03'):
        
        from_y, from_m = [int(a) for a in from_month.split('-')]
        to_y, to_m = [int(a) for a in to_month.split('-')]
    
        # dataframe container
        list_months = []
        
        if from_y==to_y:
            for m in range(from_m, to_m):
                list_months.append(f'{from_y:4d}-{m:02d}')
        else:
            for y in range(from_y, to_y):
                if y==from_y:
                    for m in range(from_m, 12+1):
                        list_months.append(f'{y:4d}-{m:02d}')
                else:
                    for m in range(1, 12+1):
                        list_months.append(f'{y:4d}-{m:02d}')
                        
            for m in range(1, to_m):
                        list_months.append(f'{to_y:4d}-{m:02d}')
       
        return list_months
    
    def load_df_from_zip(self, content: bytes):
        # convert to BytesIO for open by zipfile.
        f = BytesIO(content)
        fzip = zipfile.ZipFile(f)
        
        # check file(s) in zipfile
        file_list = fzip.namelist()
        if len(file_list)>1:
            print(f"number files in zip: {len(file_list)}")
        
        # extracted content 
        fext = BytesIO(fzip.read(file_list[0]))
        
        return self.read_klines_data(fext)
    
    
    def load_klines_spot_to_df(self, pair: str = 'BTCUSDT', 
                                interval: str = '1m', 
                                from_date: str = '2022-01-01', 
                                to_date: str = '2022-01-31'):
        
        from_date = datetime.strptime(from_date, '%Y-%m-%d')
        to_date = datetime.strptime(to_date, '%Y-%m-%d')
        
        dfs = []
        
        if from_date.year==to_date.year and from_date.month==to_date.month:
            _from_date = from_date.strftime('%Y-%m-%d')
            _to_date = to_date.strftime('%Y-%m-%d')
            list_days = self.list_daily_interval(from_date=_from_date, to_date=_to_date)
            print(list_days)
            for day in list_days:
                content = self.dl_klines_spot_daily_from_binance(pair, interval=interval, date=day)
                #content = self.load_klines_spot_daily(pair, interval=interval, date=day)
                if content==None:
                    break
                df = self.load_df_from_zip(content)
                dfs.append(df)
        
        else:
            from_date_month = from_date.replace(day=1)
            to_date_month = to_date.replace(day=1)
            _from_month = from_date_month.strftime('%Y-%m')
            _to_month = to_date_month.strftime('%Y-%m')
            list_months = self.list_monthly_interval(from_month=_from_month, to_month=_to_month)
            print(list_months)
            for month in list_months:
                content = self.dl_klines_spot_monthly_from_binance(pair, interval=interval, month=month)
                #content = self.load_klines_spot_monthly(pair, interval=interval, month=month)
                if content==None:
                    break
                df = self.load_df_from_zip(content)
                dfs.append(df)
            
            _from_date = to_date_month.strftime('%Y-%m-%d')
            _to_date = to_date.strftime('%Y-%m-%d')
            list_days = self.list_daily_interval(from_date=_from_date, to_date=_to_date)
            print(list_days)
            for day in list_days:
                content = self.dl_klines_spot_daily_from_binance(pair, interval=interval, date=day)
                #content = self.load_klines_spot_daily(pair, interval=interval, date=day)
                if content==None:
                    break
                df = self.load_df_from_zip(content)
                dfs.append(df)
                    
        # concatenate dataframes to new dataframe.
        df = pd.concat(dfs)
        
        return df


def test():
    hist = Historical_spot_data()

    pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
        'SOLUSDT', 'DOGEUSDT', 'DOTUSDT', 'TRXUSDT', 
        'MATICUSDT', 'EOSUSDT', 'BCHUSDT', 'LTCUSDT', 'ETCUSDT', 
        'FTTUSDT', 'ARBUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT',
        'NEARUSDT', 'ATOMUSDT', 'XMRUSDT', 'ALGOUSDT', 'VETUSDT',
        'XTZUSDT',
        'UNIUSDT']

    from_date = '2022-01-01'
    to_date = '2022-11-01'
    
    for pair in pairs:
        df = hist.load_klines_spot_to_df(pair=pair, interval='1m', from_date=from_date, to_date=to_date)
        print(df)
        
    
    
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--Pair", type=str, help="Pair (BTCUSDT, ETHUSDT, BNBUSDT, ...)")
    parser.add_argument("-f", "--From", type=str, help="From date")
    parser.add_argument("-t", "--To", type=str, help="To date")
    parser.add_argument("-i", "--Interval", type=str, default='1m', help="Interval")
    parser.add_argument("-o", "--Out", type=str, default=None, help="Output")
    args = parser.parse_args()
    return args
    
    
def main(args):
    hist = Historical_spot_data()
    pair = args.Pair
    from_date = args.From
    to_date = args.To
    interval = args.Interval
    
    df = hist.load_klines_spot_to_df(pair=pair, interval=interval, from_date=from_date, to_date=to_date)
    df.to_pickle(f"{pair}_{from_date}_{to_date}_{interval}.pkl.bz2")
    
    
if __name__=='__main__':
    args = options()
    main(args)
    
    """
    python binance_spot_date.py -p BTCUSDT -f 2022-01-01 -t 2024-07-01 -i 1m
    """
    #print(args.Pair)