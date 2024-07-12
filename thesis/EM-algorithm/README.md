# EM-algorithm
EM-algorithmとは未知のデータの分布を混合ガウス分布で近似する際に使用される。 <br>
詳細は以下にまとまっているので参照をお勧めする。  <br>
EMアルゴリズム徹底解説 "https://qiita.com/kenmatsu4/items/59ea3e5dfa3d4c161efb"<br>
理論自体は少しややこしいが、実装する内容は基本的に負担率の計算と負担率を用いたパラメータの更新である。  <br>

# DEMO

https://www.youtube.com/shorts/bkb0RVMxY_0



# Features

実装のポイントとしては、二次元の場合、行列の計算が複雑となるため、<br>
テンソルが使用可能なnp.einsumを使用していることである。  <br>

# Requirement
pythonでは
numpy
matplotlib
os
io
cv2
filterpy.stats
sklearn<br>
コーデックのffmpegもインストールする必要あり

# 使い方

以下の実行によりfigureフォルダの中にrandom_number~.svgとEM_algorithm.mp4が自動で作成される.<br>
random_number~.svg：データ点(赤)を生成するのに使用した分布<br>
EM_algorithm.mp4：EM-algorithmの実行途中

```bash
python EM_algorithm.py
```

# Note
maplotlibにおいて点の色を負担率毎に変えた場合、for文を用いる必要があるため、実行時間が長くなっている。



# Author

作成情報を列挙する

* 作成者：tempolin
* 所属
* E-mail：

# License
ライセンスを明示する

"EM_algorithm" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).


