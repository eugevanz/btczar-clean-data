import pandas as pd
import numpy as np
from requests import get
import time
from dotenv import load_dotenv
from os import getenv
import joblib as jb

load_dotenv()
api_key_id = getenv('LUNO_API_KEY_ID')
api_key_secret = getenv('LUNO_API_KEY_SECRET')

since = int(time.time() * 1000) - 3 * 60 * 59 * 1000
duration = 300


def get_candles(pair='XBTZAR'):
    try:
        res_json = get(
            f'https://api.luno.com/api/exchange/1/candles?pair={pair}&duration=300&since={since}',
            auth=(api_key_id, api_key_secret)
        ).json()['candles']

        candles = pd.DataFrame(res_json)
        candles.set_index('timestamp')

        candles.open = candles.open.astype(float)
        candles.close = candles.close.astype(float)
        candles.high = candles.high.astype(float)
        candles.low = candles.low.astype(float)
        candles.volume = candles.volume.astype(float)

        candles['change'] = candles.close.pct_change()

        def add_lag(df):
            for i in range(1, 4):
                candles[f'tminus_{str(i)}'] = candles.change.shift(i)

            return [f'tminus_{str(i)}' for i in range(1, 4)]

        features = add_lag(candles)

        candles['EMA12'] = candles.close.ewm(span=12).mean()
        candles.EMA12 = round(candles.EMA12)

        candles.dropna(inplace=True)

        candles['signal'] = np.where(candles.change > 0, True, False)

        model = jb.load('assets/model.pkl')

        candles['prediction'] = list(model.predict(candles[features]))

        late_close = candles.close.tolist()[-1]
        late_ema = candles.EMA12.tolist()[-1]
        late_signal = candles.prediction.tolist()[-1] and (late_ema > late_close)

        max_high = candles.high.idxmax()
        min_low = candles.low.idxmin()
        max_close = candles.close.idxmax()
        min_close = candles.close.idxmin()
        avg_close = candles.close.mean()

        return dict(candles=candles, late_signal=late_signal, late_close=late_close, late_ema=late_ema, max_high=max_high, min_low=min_low, max_close=max_close, min_close=min_close, avg_close=avg_close)
    except Exception as er:
        print(er)
