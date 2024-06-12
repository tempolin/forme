import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from filterpy.stats import plot_covariance_ellipse
from matplotlib import cm
from sklearn.cluster import KMeans
import os


#EM-algorithmを可視化するプログラム
def main():
    matplotlib.use('TkAgg')
    n_components=2 #混合数の上限
    tol = 1e-3 #emアルゴリズムの収束閾値
    max_iter = 10000 #emアルゴリズムの反復回数の上限

    #近似したい混合正規分布のパラメータを設定する
    sig = 25
    weight = np.array([0.6,0.3,0.1])#混合
    sigma = np.array([
            [[sig,0],
            [0,sig]],
            [[sig,0],
            [0,sig]],
            [[sig,0],
            [0,sig]]
    ])
    mu = np.array([
        [32, 18],
        [96, 54],
        [96, 18]
    ])

    weight_init = weight
    sigma_init = sigma
    np.random.seed(32)
    mu_init = np.empty((3,2))
    mu_init[0:2,1] = 18*np.random.randn(2)+36
    mu_init[0:2,0] = 32*np.random.randn(2)+64
    mu_init[2,0] = mu_init[1,0]+1
    mu_init[2,1] = mu_init[1,1]

    folder_name = "EMalgorithm_plot"

    # パラメータの設定
    N = 10000
    K = 3


    x1_grid = np.arange(0, 128, 1)
    x2_grid = np.arange(0, 72, 1)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    # 同時分布を計算
    P = p(X1, X2, mu, sigma, weight)
    sal = P
    # xym = np.c_[X1.ravel(), X2.ravel()]
    # N = xym.shape[0]
    xym = make_random(N,sal)
    print("km_start")
    # kmeans = KMeans(n_clusters=K).fit(xym)
    # print("km_fin")
    # mu_init = kmeans.cluster_centers_
    print(mu_init)
    N = xym.shape[0]
    H_zfill = 3

    #乱数の設定(n個)
    make_figure_gauss(X1,X2,P)
    os.makedirs(folder_name,exist_ok=True)
    sigma_plus = sigma_init
    mu_plus = mu_init
    weight_plus = weight_init
    make_figure_gaussian_plot_black_init(xym,0,H_zfill,folder_name)
    make_figure_gaussian_plot_black(xym,mu_plus,sigma_plus,1,H_zfill,folder_name)

    for i in range(2,50):
        # make_figure_gaussian_plot_black(xym, mu_plus, sigma_plus,4*i+1,H_zfill,folder_name)
        responsibility, d = cal_responsibility(xym, N, K, sigma_plus, mu_plus, weight_plus)
        make_figure_gaussian_plot_colored(xym, mu_plus,sigma_plus,4*i+2,responsibility,H_zfill,folder_name)
        gram_matrix = make_gram_matrix(d)
        sigma_plus, mu_plus, weight_plus = next_gauss(responsibility, gram_matrix, xym)
        
        make_figure_responsibility(xym, mu_plus, sigma_plus, 4*i+3, responsibility, H_zfill, folder_name)
        make_figure_gaussian_plot_colored(xym, mu_plus,sigma_plus,4*i+4,responsibility,H_zfill,folder_name)
    print(sigma_plus)
    print(mu_plus)
    print(weight_plus)
#顕著性マップの分布に従う乱数の生成
def make_random(n,sal):
    # np.random.seed(seed=32)
    max = np.max(sal)
    xym = np.empty(shape=(10000,2))
    times = 0
    while(1):
        y = np.random.randint(0, sal.shape[0])
        x = np.random.randint(0, sal.shape[1])
        z = np.random.rand()*max
        if(z < sal[y,x]):
            xym[times,0] = x
            xym[times,1] = y
            times += 1
        if(times > n-1):
            break;
    return xym

# 同時分布関数の計算
def p(x1, x2, mu, sigma, weight):
    P = 0
    # print(weight.shape[0])
    for i in range(len(weight)):
        p1 = np.exp(-np.square(x1-mu[i,0])/(2*sigma[i,0,0]))/np.sqrt(2*np.pi*sigma[i,0,0])
        p2 = np.exp(-np.square(x2-mu[i,1])/(2*sigma[i,1,1]))/np.sqrt(2*np.pi*sigma[i,1,1])
        P += (p1*p2)*weight[i]
    return P

#局面をプロットする関数
def make_figure_gauss(X1,X2,P):
    # 曲面をプロット
    fig = plt.figure(figsize = (8, 8))
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    fig.patch.set_edgecolor('white')
    # plt.style.use('ggplot')
    plt.style.use('default')
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x", size = 16)  # x1軸
    ax.set_ylabel("y", size = 16)  # x2軸
    ax.set_zlabel("z", size = 16)  # p軸

    # ax.plot_surface(X1, X2, P, cmap = "YlGn_r")
    ax.plot_surface(X1, X2, P,cmap=cm.coolwarm,alpha=0.4)
    # ax.scatter()
    sal = P
    max = np.max(sal)

    xy_in = np.empty(shape=(10000,3))
    xy_out = np.empty(shape=(10000000, 3))
    time_in = 0
    time_out = 0
    n = xy_in.shape[0]
    np.random.seed(seed=32)
    while(1):
        y = np.random.randint(0, 128)
        x = np.random.randint(0, 72)
        z = np.random.rand()*max
        if(z < sal[x,y]):
            xy_in[time_in,0] = x
            xy_in[time_in,1] = y
            xy_in[time_in, 2] = z
            time_in += 1
        else:
            xy_out[time_out,0] = x
            xy_out[time_out,1] = y
            xy_out[time_out, 2] = z
            time_out += 1
        if(time_in > n-1):
            break
    K=1000
    xy_in = xy_in[0:K,:]
    xy_out = xy_out[0:K, :]
    ax.invert_yaxis()
    ax.scatter(xy_in[:,1], xy_in[:,0], xy_in[:,2], marker=".", color="red")
    ax.scatter(xy_out[:,1], xy_out[:,0], xy_out[:,2], marker=".", color="green")
    plt.savefig("figure/random_based_gauss.svg")

    # plt.show()
    plt.close(fig)

#負担率，ガウス中心座標の計算
def cal_responsibility(xym, N, K, sigma, mu, weight):
    # x座標の設定，無理やりN×K×1×2にしている
    x_all = np.repeat(xym[:, np.newaxis, :], K, axis=1)
    x_all = np.repeat(x_all[:, :, np.newaxis, :], 1, axis=2)
    #平均muの設定，無理やりN×K×1×2にしている
    # print(mu.shape)
    mu_all = np.tile(mu, (N, 1, 1))
    mu_all = np.repeat(mu_all[:,:,np.newaxis,:],1,axis=2)
    # print(mu_all.shape)

    #分散共分散行列の設定，N×K×2×2行列
    sigma_all = np.tile(sigma, (N, 1, 1, 1))

    det = np.linalg.det(sigma_all[:])
    inv = np.linalg.inv(sigma_all)
    d = x_all - mu_all
    n = x_all.shape[1]
    # print(x_all.shape[1])
    norm = np.sqrt((2 * np.pi) ** n * det)
    power = - np.einsum('ijkl, ijkl,ijkl->ij',d, inv, d) /2.0

    gauss = np.exp(power)/norm
    gaussian_one = weight*gauss

    gaussian_mixture = np.sum(gaussian_one,axis=1)
    gaussian_mixture = np.sum(gaussian_one,axis=1)[:,np.newaxis]
    responsibility = gaussian_one/gaussian_mixture
    responsibility = responsibility/np.sum(responsibility,axis=1)[:,np.newaxis]
    return responsibility, d

def make_gram_matrix(d):
    #１×2行列の1部分を取り出している．
    d_p = d[:, :, 0 ,:]
    gram_matrix = d_p[:, :, :, np.newaxis] * d_p[:, :, np.newaxis, :]
    return gram_matrix

def next_gauss(responsibility, gram_matrix, xym):
    #分散共分散行列の更新
    res = responsibility[:,:,np.newaxis,np.newaxis]
    sigma_plus_top = res*gram_matrix
    sigma_plus_top = np.sum(sigma_plus_top,axis=0)
    sigma_plus_bottom = np.sum(res,axis=0)
    sigma_plus = sigma_plus_top/sigma_plus_bottom

    #平均の更新
    mu_plus_top = responsibility.T @ xym
    mu_plus_bottom = np.sum(responsibility,axis=0)[:,np.newaxis]
    mu_plus = mu_plus_top/mu_plus_bottom

    weight_plus_top = np.sum(responsibility,axis=0)
    weight_plus_bottom = np.sum(responsibility)
    weight_plus = weight_plus_top/weight_plus_bottom

    return sigma_plus, mu_plus, weight_plus

def make_figure_responsibility(xym, mu, sigma, K, responsibility, H_zfill, folder_name):
    fc2 = ["r", "g", "b"]
    # グラフ表示
    fig, ax = plt.subplots(1,1,figsize=(12.8,7.2))
    for i in range(responsibility.shape[0]):
        ax.plot(xym[i,0],xym[i,1],marker=".",linewidth=0,color=(responsibility[i,0],responsibility[i,1],responsibility[i,2],1),zorder=0)
        # print(i)
    ax.invert_yaxis()

    # for i in range(mu.shape[0]):
    #     a = plot_covariance_ellipse(mu[i,:], sigma[i,:], fc=fc2[i], alpha=0.3, std=[1,2,3])
    
    ax.set_xlim(0,128)
    ax.set_ylim(0,72)
    # name = folder_name+"/"+str(K).zfill(H_zfill)+".svg"
    # fig.savefig(name)
    name = folder_name+"/"+str(K).zfill(H_zfill)+".png"
    fig.savefig(name)
    plt.close(fig)

def make_figure_gaussian_plot_colored(xym, mu, sigma, K, responsibility, H_zfill, folder_name):
    # グラフ表示
    fc2 = ["r", "g", "b"]
    fig, ax = plt.subplots(1,1,figsize=(12.8,7.2))
    for i in range(responsibility.shape[0]):
        ax.plot(xym[i,0],xym[i,1],marker=".",linewidth=0,color=(responsibility[i,0],responsibility[i,1],responsibility[i,2],0.3),zorder=0)

    for i in range(mu.shape[0]):
        a = plot_covariance_ellipse(mu[i,:], sigma[i,:], fc=fc2[i], alpha=0.3, std=[1,2,3])
    plt.close(fig)
    ax.set_xlim(0,128)
    ax.set_ylim(0,72)
    name = folder_name+"/"+str(K).zfill(H_zfill)+".png"
    fig.savefig(name)
    plt.close(fig)  

def make_figure_gaussian_plot_black(xym, mu, sigma, K, H_zfill, folder_name):
    # グラフ表示
    fc2 = ["r", "g", "b"]
    fig = plt.figure(figsize=(12.8,7.2*2))
    ax = plt.subplots(1,1,figsize=(12.8,7.2))
    ax.plot(xym[:,0],xym[:,1],marker=".",linewidth=0,color="k",zorder=0)
    ax.invert_yaxis()
    ax.set_xlim(0,128)
    ax.set_ylim(0,72)
    for i in range(mu.shape[0]):
        a = plot_covariance_ellipse(mu[i,:], sigma[i,:], fc=fc2[i], alpha=0.3, std=[1,2,3])
    plt.close(fig)
    print(mu)
    name = folder_name+"/"+str(K).zfill(H_zfill)+".png"
    fig.savefig(name)
    plt.close(fig)    

def make_figure_gaussian_plot_black_init(xym, K, H_zfill, folder_name):
    # グラフ表示
    fc2 = ["r", "g", "b"]
    fig, ax = plt.subplots(1,1,figsize=(12.8,7.2))
    ax.plot(xym[:,0],xym[:,1],marker=".",linewidth=0,color="k",zorder=0)
        # print(i)
    ax.invert_yaxis()
    plt.close(fig)
    ax.set_xlim(0,128)
    ax.set_ylim(0,72)
    # print(mu)
    # name = folder_name+"/"+str(K).zfill(H_zfill)+".svg"
    # fig.savefig(name)
    name = folder_name+"/"+str(K).zfill(H_zfill)+".png"
    fig.savefig(name)
    plt.close(fig)  


if __name__ == "__main__":
    main()
