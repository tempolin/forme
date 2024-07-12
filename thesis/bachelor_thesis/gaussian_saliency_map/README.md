# gaussian_saliency_map
本プログラムの内容は既存の顕著性マップをガウス分布を用いて近似することである<br>
顕著性マップを近似する理由は推定に用いる場合，顕著性マップのパラメータ数が非常に多いからである．<br>
顕著性マップを近似することで顕著性マップ1枚当たりのパラメータ数が1920×1080→6×Kに減少する．(K:混合数)<br>

# DEMO
以下に示すものは顕著性マップ(左)と混合ガウス分布を用いた顕著性マップの近似結果(右)である．<br>
動画の顕著性が存在する領域に対して大幅に情報を失っていないことが分かる。<br>
|Before|After|
|:-:|:-:|
|<video src="SaliencyMapExample01.mp4">|<video src="GaussianSaliencyMapExample01.mp4">|
<div><video controls src="<!---"SaliencyMapExample01.mp4"--->" muted="false"></video></div>
# Requirement

"hoge"を動かすのに必要なライブラリなどを列挙する
cv2、sys、numpy、sklearn.mixture、matplotlib.pyplotなど
# Installation
# Usage
動画(Sample.mp4)を用意する
以下のコマンドで動画の顕著性マップ、顕著性マップの近似結果、近似に使用したデータ点の動画が保存される。

```bash
python MakeGaussianSaliencyMap.py Sample.mp4
```

# Note
混合ガウス分布を計算+動画を作成しているので計算時間が非常に長い()<br>
今回，デフォルトの顕著性マップとしてOpenCVの顕著性マップを用いている<br>
提案手法のような精度が必要であれば，Drosteの顕著性マップ(再配布できないため)をsalに代入する必要がある．<br>


# Author

作成情報を列挙する

* 作成者：tempolin
* 所属
* E-mail：

# License
under [MIT license](https://en.wikipedia.org/wiki/MIT_License).


