import requests
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# APIのURL
url = "https://archive-api.open-meteo.com/v1/archive"

# データとして用いる都市
cities = {#"Tokyo": (35.6895,139.6917),
          "Osaka": (34.6856, 135.5286),
          "Nagoya": (35.185033, 136.899946),
          "Okayama": (34.667384, 133.934447),
          "Himeji": (34.837768, 134.693678),
          "Hiroshima": (34.401292, 132.459853)}

# 予測の対象
targetCity = "Nagoya"

data={}

for city,lonlat in cities.items():
    params = {
        "latitude": lonlat[0],  # 緯度
        "longitude": lonlat[1], # 経度
        "start_date": "2013-01-01",
        "end_date": "2025-12-31",
        "daily": ["temperature_2m_mean", "precipitation_sum"], # 降水量
        "timezone": "Asia/Tokyo"
    }
    response = requests.get(url, params=params)
    data[city] = response.json()

# 各都市のデータを格納するリスト
df_list = []

for city, raw_data in data.items():
    # 各都市の「日付」と「平均気温」を取り出してDataFrame化
    temp_df = pd.DataFrame({
        "date": raw_data["daily"]["time"],
        f"{city}_precip": raw_data["daily"]["precipitation_sum"],
        f"{city}_temp": raw_data["daily"]["temperature_2m_mean"]
    })
    
    # 日付をインデックス（行ラベル）に設定
    temp_df.set_index("date", inplace=True)
    df_list.append(temp_df)

# 全てのDataFrameを列方向に結合
final_df = pd.concat(df_list, axis=1)

for city in cities.keys():
    final_df[f'{city}_precip_prev1'] = final_df[f'{city}_precip'].shift(1)
    final_df[f'{city}_temp_prev1'] = final_df[f'{city}_temp'].shift(1)
final_df['Nagoya_temp_diff'] = final_df['Nagoya_temp'].diff()
final_df['Nagoya_precip_ma3'] = final_df['Nagoya_precip'].rolling(window=3).mean().shift(1)

# shiftや移動平均を使うと最初の数行に欠損値(NaN)が出るので削除
final_df = final_df.dropna()
final_df = final_df.sort_index()

#学習用データ
train_df = final_df.loc[:"2022-12-31"]
#テスト用データ
test_df = final_df.loc["2023-01-01":]

model = LinearRegression()

#パラメータを列挙
param_list=['Nagoya_temp_prev1','Nagoya_temp_diff','Nagoya_precip_ma3']
for city in cities.keys():
    param_list.append(f'{city}_precip_prev1')

# 目的変数 y (名古屋の降水量)
y = train_df[f'{targetCity}_precip']

# 説明変数 X (名古屋以外の都市と、自分で作った特徴量)
X = train_df[param_list]

# 学習プロセス
model.fit(X, y)

print(pd.DataFrame({"Name":X.columns,
                    "Coefficients":model.coef_}).sort_values(by='Coefficients') )

X_test = test_df[param_list]
y_test = test_df[f'{targetCity}_precip']

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'決定係数 (R2): {r2:.3f}')
print(f'平均絶対誤差 (MAE): {mae:.3f} 度')