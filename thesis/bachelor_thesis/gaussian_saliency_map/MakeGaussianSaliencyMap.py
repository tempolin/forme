import cv2
import sys
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt
import time
import os
import glob
'''
講義動画を受け取り，顕著性マップの動画と混合ガウス分布で近似した動画を表示するプログラム
オプションでnpzファイルを作る方法も追加するかもしれない

'''

def main():
    #動画のパスを設定 python MakeGaussianSaliencyMap [Sample.mp4]
    if(sys.argv[1]):#動画のパスが指定されているか確認する
        video_name = sys.argv[1]
    else:
        print("動画ファイルが指定されていません")
        sys.exit()

    cap = cv2.VideoCapture(video_name)#
    if not cap.isOpened():#動画の存在を確認する
        print("Could not open video")
        sys.exit()

    #顕著性マップを作成したい動画の設定を取得する    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))#フレーム数
    fps = (cap.get(cv2.CAP_PROP_FPS))#フレームレート
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))#横
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))#縦

    #動画のコーデックの指定(mp4を指定している)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    #ウィンドウ名の指定＋mp4ファイルを作成するvideowriterの設定(mp4ファイルで元の動画の設定で指定)

    #元の動画
    window_name_video = "video" 

    #顕著性マップ
    window_name_sal = "SaliencyMap"
    video_saliency_name = "SaliencyMapExample01.mp4"
    video_saliency = cv2.VideoWriter(video_saliency_name, fourcc,fps, (w,h))

    #顕著性マップの近似に使用するデータ
    window_name_gauss_sal = "GaussianSaliencyMap"
    video_gauss_saliency_name = "GaussianSaliencyMap.mp4"
    video_gauss_saliency = cv2.VideoWriter(video_gauss_saliency_name, fourcc,fps, (w,h))

    #顕著性マップの近似を表示する
    window_name_random = "RandomPointSaliencyMap"
    video_name_random = "RandomPointSaliencyMap.mp4"
    video_random = cv2.VideoWriter(video_name_random, fourcc, fps, (w,h))

    #描画するウィンドウの大きさの設定
    ws = 128*3
    hs = 72*3

    #顕著性マップの近似結果を計算するためのw×h座標作成
    x = np.arange(0, w)
    y = np.arange(0, h)
    X, Y = np.meshgrid(x, y)
    XY = np.c_[X.ravel(), Y.ravel()]

    #顕著性マップの近似と描画を行う関数
    for i in range(0, n_frames, 1):
        ret, frame = cap.read()#動画を読み込む
        if ret == False:
            print("could not get frame")
            continue

        #saliencyマップの作成　Open-cv内の関数の呼び出し(opencv-contrib_pythonのインストール必須
        #他の顕著性マップを使用する場合はsalを変更すればよい
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(frame)
        sal = (saliencyMap*255).astype("uint8")

        # #顕著性マップに従うデータ点の作成(n=50000個と設定)
        n = 50000
        xy_random = make_random(n,sal)
        
        #データ点を用いたEMアルゴリズムの実装：https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture
        n_components=10 #混合数の上限
        tol = 1e-3 #emアルゴリズムの収束閾値
        max_iter = 10000#emアルゴリズムの反復回数の上限
        covariance_type='full'#使用するガウス分布の分散共分散行列の形、fullなら一般の行列
        results_bayes = GaussianMixture(
            n_components=n_components,covariance_type=covariance_type,tol=tol,max_iter=max_iter
            ).fit(xy_random)#混合ガウス分布の推定
        # results_bayes = BayesianGaussianMixture(
        #     n_components=n_components,covariance_type=covariance_type,tol=tol,max_iter=max_iter
        #     ).fit(xy_random)#混合ガウス分布の推定


        #EMアルゴリズムの結果を取得
        weight = results_bayes.weights_#ガウス分布の重み
        means = results_bayes.means_#ガウス分布の平均値
        covariances = results_bayes.covariances_#ガウス分布の分散共分散行列
        lower_bound = results_bayes.lower_bound_#emアルゴリズムの下限
        n_iter = results_bayes.n_iter_#emアルゴリズムの反復回数

        #近似結果を描画するために計算する
        gauss_sal = gaussian_preprocessing(XY,weight,means,covariances)
        Z_min = np.min(gauss_sal)
        Z_max = np.max(gauss_sal)
        gauss_sal = gauss_sal.reshape(X.shape)
        gauss_sal = (gauss_sal-Z_min)*255/(Z_max-Z_min)
        gauss_sal = gauss_sal.astype("uint8")

        #動画の表示
        cv2.imshow(window_name_video, cv2.resize(frame, (ws, hs)))
        cv2.moveWindow(window_name_video,0,0)
        key = cv2.waitKey(1)
        if(key == 27):
            break
        
        #顕著性マップの表示、保存
        sal =  cv2.applyColorMap(sal, cv2.COLORMAP_JET)
        video_saliency.write(sal)
        cv2.imshow(window_name_sal, cv2.resize(sal, (ws, hs)))
        cv2.moveWindow(window_name_sal, ws,0)
        key = cv2.waitKey(1)
        if(key == 27):
            break

        #顕著性マップの近似に使用したデータ点の表示、保存
        img = np.zeros((h, w, 3)).astype("uint8")
        xy_random = xy_random.astype(np.int64)
        for i in range(xy_random.shape[0]):
            cv2.circle(img, (xy_random[i,0], xy_random[i,1]), 1, (255,255,255), -1)
        video_random.write(img)
        cv2.imshow(window_name_random, cv2.resize(img, (ws,hs)))
        cv2.moveWindow(window_name_random, ws*2,0)
        key = cv2.waitKey(1)
        if(key == 27):
            break
        
        #近似した県都性マップの表示、保存
        gauss_sal = cv2.applyColorMap(gauss_sal,cv2.COLORMAP_JET)
        video_gauss_saliency.write(gauss_sal)
        cv2.imshow(window_name_gauss_sal, cv2.resize(gauss_sal, (ws, hs)))
        cv2.moveWindow(window_name_gauss_sal, ws*3,0)
        key = cv2.waitKey(1)
        if(key == 27):
            break

        # np_pass = str(i).zfill(len(str(len(files))))
        # np_pass = os.path.join(npz_folder_path,np_pass)+".npz"
        # np.savez_compressed(np_pass,img=img,sal=sal,xy_random=xy_random,weight=weight, means=means, covariances=covariances)
        # print(np_pass)
    #動画の保存
    video_random.release()
    video_gauss_saliency.release()
    video_saliency.release()
 
    #表示に使用したウィンドウの破壊
    cv2.destroyWindow(window_name_video)
    cv2.destroyWindow(window_name_sal)
    cv2.destroyWindow(window_name_gauss_sal)
    cv2.destroyWindow(window_name_random)


#二次元ガウス分布の計算
def gaussian2(x, mu, sigma):
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    d = x - mu
    n = x.ndim
    norm = np.sqrt((2 * np.pi) ** n * det)
    power = - np.einsum('il,lk,ik->i', d, inv, d) /2.0
    # print(power)
    return np.exp(power) / norm

#混合ガウス分布にx座標とy座標を代入した場合の値を計算(二次元にしか対応していない)
def gaussian_preprocessing(XY, weight, means , covariances):
    Z = 0
    for j in range(len(weight)):
       Z += weight[j]*gaussian2(XY, means[j], covariances[j])
    return Z

#顕著性マップの分布に従うデータ点の生成
def make_random(n,sal):
    np.random.seed(16)
    max = np.max(sal)
    xym = np.empty(shape=(int(n),2))#二次元の乱数を格納する配列
    times = 0
    # print(sal.shape[1],sal.shape[0])
    while(1):
        y = np.random.randint(0, sal.shape[0])
        x = np.random.randint(0, sal.shape[1])
        z = np.random.rand()*max
        # print(z, sal[y,x])
        if(z < sal[y,x]):
            xym[times,0] = x
            xym[times,1] = y
            times += 1
        if(times > n-1):
            break;
    return xym

# #ガウシアンの計算をしている
# def gaussian_mixture(X_t, weight, means, covariances, K):
#     # x座標の設定，無理やりN×K×1×2にしている
#     N = X_t.shape[0]
#     x_all = np.repeat(X_t[:, np.newaxis, :], K, axis=1)
#     # x_all = np.repeat(x_all[:, :, np.newaxis, :], 1, axis=2)
#     #平均muの設定，無理やりN×K×1×2にしている
#     mu_all = np.tile(means, (N, 1, 1))
#     # mu_all = np.repeat(mu_all[:,:,np.newaxis,:],1,axis=2)

#     #分散共分散行列の設定，N×K×2×2行列
#     sigma_all = np.tile(covariances, (N, 1, 1, 1))
#     print(sigma_all.shape)
#     det = np.linalg.det(sigma_all[:])
#     inv = np.linalg.inv(sigma_all)
#     d = x_all - mu_all
#     D = x_all.shape[2]
#     norm = np.sqrt((2 * np.pi) ** D * det)
#     # power = - np.einsum('ijkl, ijkl,ijkl->ij',d, inv, d) /2.0
#     # power = - np.einsum('ijk, ijkl,ijl->ij',d, inv, d,optimize='optimal') /2.0
#     # print(np.einsum("abk,cdkl -> l"))
#     power =  -np.einsum('ijk, ijkl,ijl->ij',d, inv, d,optimize='optimal') /2.0
#     gauss = np.exp(power)/norm
#     weight = np.tile(weight, (N, 1))
#     weighted_gauss = np.einsum("ij,ij->i",weight,gauss,)
#     return weighted_gauss

if __name__ == "__main__":
    main()
