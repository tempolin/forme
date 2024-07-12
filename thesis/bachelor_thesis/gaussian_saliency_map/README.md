# gaussian_saliency_map
本プログラムの内容は既存の顕著性マップをガウス分布を用いて近似することである<br>
顕著性マップを近似する理由は推定に用いる場合，顕著性マップのパラメータ数が非常に多いからである．<br>
顕著性マップを近似することでパラメータ数が1920×1080→60に減少する．<br>

# DEMO
以下に示すものは顕著性マップ(左)と混合ガウス分布を用いた顕著性マップの近似結果(右)である．<br>
動画の顕著性が存在する領域に対して大幅に情報を失っていないことわかる．<br>
![SaliencyMapExample01](https://github.com/tempolin/forme/assets/168509729/a9486594-9611-419d-aa3c-92263abc3a70)
![GaussianSaliencyMapExample01](https://github.com/tempolin/forme/assets/168509729/5644359d-ad6d-4ffd-bae0-c94ef5bb77e8)<br>

# Requirement

"hoge"を動かすのに必要なライブラリなどを列挙する

* huga 3.5.2
* hogehuga 1.0.2

# Installation
# Usage
動画(Sample.mp4)を用意する

```bash
python Gaussian.py Sample.mp4
```

# Note
今回，デフォルトの顕著性マップとしてOpenCVの顕著性マップを用いている<br>
提案手法のような精度が必要であれば，Drosteの顕著性マップ(再配布できないため)をsalに代入する必要がある．<br>

# Author

作成情報を列挙する

* 作成者：tempolin
* 所属
* E-mail：

# License
under [MIT license](https://en.wikipedia.org/wiki/MIT_License).


