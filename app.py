from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import io
from scipy.optimize import fsolve
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/draw_circle', methods=['POST'])
def draw_circle():
    radius = float(request.form.get("zigzag_divnum", 5))  # POSTリクエストで半径を取得

    # 円を描画
    fig, ax = plt.subplots()
    circle = plt.Circle((0, 0), radius, color='blue', fill=False)
    ax.add_patch(circle)
    ax.set_xlim(-radius-1, radius+1)
    ax.set_ylim(-radius-1, radius+1)
    ax.set_aspect('equal')

    # 画像をメモリ上で処理
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    
    return send_file(buf, mimetype="image/png")  # 画像をレスポンスとして送信

@app.route('/divide_circle', methods=['POST'])
def divide_circle():
    # ケーキを何等分するか(n: 3以上の分割数)
    n = float(request.form.get("zigzag_divnum", 5))  # POSTリクエストで円の分割数を取得

    # i行j列のデータフレームを作成するメソッド
    # 第3引数には可変長引数を使用し、カラム名をj個指定できる（j個以外のカラム数を指定するとValueErrorになる）
    def make_df(i, j, *args):
        df = pd.DataFrame([[0]*j for k in range(i)])
        df = df.set_axis(list(args),axis='columns')
        return df

    # sin, cos, tanのメソッドを定義（コードを書きやすくするため）
    def sin(theta):
        return np.sin(theta)
    def cos(theta):
        return np.cos(theta)
    def tan(theta):
        return np.tan(theta)
    # 円の半径
    r = 1

    # 円の中心から最も近い円周角に関する式を定義
    # 分割数の偶奇によって立てる式が異なる
    def angle_initial(theta):
        if n % 2 == 1:
            return sin(theta) + theta - np.pi/n
        else:
            return (1/2)*sin(2*theta) + theta - np.pi/n

    # 一番端の弦からなる円周角phiに関する式
    def angle_phi(phi):
        return phi - (1/2)*sin(2*phi) - np.pi/n

    # 円周角thetaの円周
    def calc_circle(theta):
        return 2*r*theta
    # 円周角thetaの弦の長さ
    def calc_chord(theta):
        return 2*r*sin(theta)
    # radianからdegreeに変換するメソッド
    def convert_rad_to_deg(theta):
        return np.degrees(theta)

    # 格納しておく角度の変数の数n
    # 半円のうち、円の中心から最も近い領域(theta_init)と最も遠い領域(phi)を除いた領域(REST)の数m
    if n % 2 == 1:
        n_theta = int((n-1)/2)
        m = int((n-3)/2)
    else:
        n_theta = int((n-2)/2)
        m = int((n-4)/2)

    # 円周角と、その円周角に対する弦および周の長さを格納する配列を定義
    # 1列目：theta、2列目：弦、3列目：周になる
    try:
        df_theta = make_df(n_theta, 6, 'θ[radian]', 'θ[degree]', 'chord(θ)', 'circle(θ)', 'sin(θ)', 'cos(θ)')
    except ValueError:
        print('第2引数と同じ数だけのカラム名を、タプル（カンマで区切った要素）で指定してください')
    #df_theta = pd.DataFrame([[0]*6 for i in range(n_theta)]).set_axis(['θ[radian]', 'θ[degree]', 'chord(θ)', 'circle(θ)', 'sin(θ)', 'cos(θ)'], axis='columns')

    # i番目の角度θが分かったらこのメソッドを実行する
    def set_object(i, theta):
        df_theta.loc[i,['θ[radian]']] = theta
        df_theta.loc[i,['θ[degree]']] = convert_rad_to_deg(df_theta['θ[radian]'][i])
        df_theta.loc[i,['chord(θ)']] = calc_chord(df_theta['θ[radian]'][i])
        df_theta.loc[i,['circle(θ)']] = calc_circle(df_theta['θ[radian]'][i])
        df_theta.loc[i,['sin(θ)']] = sin(df_theta['θ[radian]'][i])
        df_theta.loc[i,['cos(θ)']] = cos(df_theta['θ[radian]'][i])

    # df_thetaのゲッターを定義
    def get_angle(i):
        return df_theta['θ[radian]'][i]
    def get_angle_degree(i):
        return df_theta['θ[degree]'][i]
    def get_chord(i):
        return df_theta['chord(θ)'][i]
    def get_circle(i):
        return df_theta['circle(θ)'][i]
    def get_sin(i):
        return df_theta['sin(θ)'][i]
    def get_cos(i):
        return df_theta['cos(θ)'][i]
    def get_object(i):
        get_angle(i)
        get_angle_degree(i)
        get_chord(i)
        get_circle(i)
        get_sin(i)
        get_cos(i)
        return df_theta.loc[i,:]

    # 全ての角度を足すメソッド
    def sum_degrees():
        if n % 2 == 0:
            return int(round(4*(df_theta['θ[degree]'].sum()+phi_deg)))
        else:
            return int(round(4*(df_theta['θ[degree]'].sum()+phi_deg)-2*df_theta['θ[degree]'][0]))
    def sum_radians():
        if n % 2 == 0:
            return int(round(4*(df_theta['θ[radian]'].sum()+phi)))
        else:
            return int(round(4*(df_theta['θ[radian]'].sum()+phi)-2*df_theta['θ[radian]'][0]))

    # 取り敢えず、thetaとphiを求めておく　初期値をπ/(n+6)として計算する
    # これらの値を求めるメソッドは、何分割しようと変わらない
    theta_init = fsolve(angle_initial, np.pi/(n+6))[0]
    phi = fsolve(angle_phi, np.pi/6)[0]
    # 0番目の角度thetaが分かったので各値を挿入
    set_object(0, theta_init)

    # 角度αを定義
    if n % 2 == 0:
        alpha = np.pi/2 - theta_init
    else:
        alpha = (np.pi-theta_init)/2
    # 使うか分からないが、phiとtheta_initの単位をdegreeに変換したものを新たな変数に代入
    phi_deg = convert_rad_to_deg(phi)
    # theta_init_deg = convert_rad_to_deg(phi)

    # 領域RESTが存在する場合の処理
    if m == 1:
        # 領域RESTをm分割したときの円周角の合計γ
        if n % 2 == 1:
            gamma = np.pi/2 - theta_init/2 - phi
        else:
            gamma = np.pi/2 - theta_init - phi
        set_object(1, gamma)
    elif m > 1:
        # theta_initの円周角を構成する1辺の長さe
        if n % 2 == 1:
            e = 2*r*cos(theta_init/2)
        else:
            e = 2*r*cos(theta_init)
            
        # 一番端の弦
        chord_phi = calc_chord(phi)
        # 円周角theta_iの数列と、それを構成する1辺の長さxの配列(数列)
        try:
            df_seq = make_df(n_theta, 2, 'θ[radian]','chord_x(θ)')
        except ValueError:
            print('第2引数と同じ数だけのカラム名を、タプル（カンマで区切った要素）で指定してください')

        # このデータフレームに関するゲッターとセッターを定義
        def set_angle_x(i, theta):
            df_seq.loc[i,['θ[radian]']] = theta
            df_seq.loc[i,['chord_x(θ)']] = calc_chord(theta)
        def get_x_seq(i):
            return df_seq['chord_x(θ)'][i]
        def get_angle_seq(i):
            return df_seq['θ[radian]'][i]
        # df_seqの0番目に、それぞれ値を代入
        set_angle_x(0, alpha)
        
        # ここから、各xおよびthetaを求めていく
        for i in range(1, n_theta):
            def f(theta):
                return get_angle_seq(i-1) - theta
            # def x(theta):
            #     return 2*r*sin(theta_dash(theta))
            # 面積に関する式:（円周角thetaから成る領域の面積S）-（円の面積の1/n倍）
            def part_square(theta):
                S = (1/2)*get_x_seq(i-1)*calc_chord(f(theta))*sin(theta) + r**2*(theta-(1/2)*sin(2*theta))
                return S - r**2*np.pi/n
            
            # i番目のpart_suquareが0になる円周角theta_iを計算
            # fsolveの初期値は、π*i/nとした
            theta_i = fsolve(part_square, np.pi*i/n)[0]
            # theta_iをdf_thetaに格納
            set_object(i, theta_i)
            # theta_iとx(theta_i)を、作成しておいたデータフレームにセット
            set_angle_x(i, f(theta_i))
            
    # 全ての角度を足して、360°になってるかを確認する
    # print('1周分の角度:',str(sum_degrees())+'°')

    # print(df_theta)
    # print('φ =',round(phi_deg,5),'[degree]')

    x_range = np.arange(0,n_theta,1)
    #plt.figure(figsize=(24, 18))
    plt.scatter(x_range, df_theta.loc[:,['θ[degree]']] , color="blue", label='k vs theta_k(degrees)')
    plt.xticks(x_range)
    plt.xlabel('k')
    plt.ylabel('theta_k(degree)')
    plt.title('Relationship between number of divisions and theta_k where n = '+str(n))
    plt.grid(False)
    plt.legend()
    plt.savefig('n='+str(n)+'.png', transparent=False) #必ずplt.show()の前にこれを持ってくること。
    plt.show()

    # ここから、全ての円周上の座標を求めて線を引いていく
    # それぞれの座標を格納するデータフレーム
    try:
        df_P = make_df(n_theta+1, 3, 'θ[radian]', 'x(θ)', 'y(θ)') # 第一、第四象限
        df_P_dash = make_df(n_theta+1, 3, 'θ[radian]', 'x(θ)', 'y(θ)') # 第二、第三象限
    except ValueError:
        print('第2引数と同じ数だけのカラム名を、タプル（カンマで区切った要素）で指定してください')
    # 上記のデータフレームのx, y座標を取り出すメソッド（ゲッター）
    def get_x_coord(df, i):
        return df['x(θ)'][i]
    def get_y_coord(df, i):
        return df['y(θ)'][i]

    # 角度と座標を挿入するメソッド
    def insert_object_into_P(i, theta):
        df_P.loc[i,['θ[radian]']] = theta
        df_P_dash.loc[i,['θ[radian]']] = theta
        if n % 2 == 1:
            if i % 2 == 0:
                df_P.loc[i,['x(θ)']] = r*sin(theta)
                df_P.loc[i,['y(θ)']] = r*cos(theta)
                df_P_dash.loc[i,['x(θ)']] = -r*sin(theta)
                df_P_dash.loc[i,['y(θ)']] = r*cos(theta)
            else:
                df_P.loc[i,['x(θ)']] = r*sin(theta)
                df_P.loc[i,['y(θ)']] = -r*cos(theta)
                df_P_dash.loc[i,['x(θ)']] = -r*sin(theta)
                df_P_dash.loc[i,['y(θ)']] = -r*cos(theta)
        else:
            if i % 2 == 0:
                df_P.loc[i,['x(θ)']] = r*sin(theta)
                df_P.loc[i,['y(θ)']] = -r*cos(theta)
                df_P_dash.loc[i,['x(θ)']] = -r*sin(theta)
                df_P_dash.loc[i,['y(θ)']] = r*cos(theta)
            else:
                df_P.loc[i,['x(θ)']] = r*sin(theta)
                df_P.loc[i,['y(θ)']] = r*cos(theta)
                df_P_dash.loc[i,['x(θ)']] = -r*sin(theta)
                df_P_dash.loc[i,['y(θ)']] = -r*cos(theta)

    # 数列の初項
    # nが奇数なら1番目に入るのはtheta_init、偶数なら2*theta_init
    if n % 2 == 1:
        a_0 = theta_init
    else:
        a_0 = 2*theta_init
    # df_P, df_P_dashともに0番目は0が入る
    insert_object_into_P(0, 0)
    insert_object_into_P(1, a_0)

    # 分割数nが5以上になるとP, P_dashにさらに値を入れていかなくてはいけない、
    if n >= 5:
        # df_thetaの偶奇の添え字の角度だけを足すメソッド
        # データフレームの2番目または1番目からi番目までの要素を1個飛ばしで見て足し上げる（結果的に偶数番目の添え字の要素だけを足し算することになる）
        def sum_angle(df, i):
            if i % 2 == 1:
                return a_0 + 2*df['θ[radian]'].iloc[2:i:2].sum() # iloc[start:end:stop]
            else:
                return 2*df['θ[radian]'].iloc[1:i:2].sum()
        
        for k in range(2, n_theta+1):
            insert_object_into_P(k, sum_angle(df_theta, k))

    print(df_P)
    print(df_P_dash)

    fig, ax = plt.subplots()
    circle = plt.Circle((0,0), r, color='purple', fill=False)
    ax.add_artist(circle)
    # 線を引くメソッドを定義 引数の説明(pandas.DataFrame, str)
    def draw_line(df, color):
        # 線の数=点の数-1
        n_line = len(df)-1
        for i in range(n_line):
            ax.plot([get_x_coord(df,i), get_x_coord(df,i+1)],
                    [get_y_coord(df,i), get_y_coord(df,i+1)], 
                    color=color)

    draw_line(df_P, 'red')
    draw_line(df_P_dash, 'green')
    if n % 2 == 0:
        ax.plot([0,0],[-1,1], color='blue')
    ax.set_aspect('equal')
    ax.set_xlim(-r-0.5, r+0.5)
    ax.set_ylim(-r-0.5, r+0.5)
    ax.grid(True)

    # 画像をメモリ上で処理
    buf = io.BytesIO()
    fig.savefig(buf, format="png")  # `plt` ではなく `fig` を指定
    buf.seek(0)
    
    return send_file(buf, mimetype="image/png")  # 画像をレスポンスとして送信

if __name__ == '__main__':
    app.run(debug=True)