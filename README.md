# camellia 2024
## data_utils.py
example:

## indicators.py
example:
```python
> from indicators import simple_moving_average_tf_delay
> df_full = load_from_linode_pickle_bz2(f"ETHUSDT_2022_1m", "price_data")
> timefilter = ((df_full.index>='2023-07-01')&(df_full.index<'2024-01-01'))
> df = df_full.loc[timefilter].copy()
> price = df[['open','high','low','close']]
> t_1m = (price.index.astype(np.int64) // 10 ** 9 // 60).values
> p_1m = price.values
> ma = simple_moving_average_tf_delay(t_1m, p_1m, 60, 24)
```

## binance_spot_data.py
example:
```bash
> python binance_spot_data.py -p BTCUSDT -f 2018-01-01 -t 2024-07-01 -i 1m
```

loading:
```python
> import pandas as pd
> df = pd.read_pickle("BTCUSDT_2018-01-01_2024-07-01_1m.pkl.bz2")
> df[["close"]].plot()
```