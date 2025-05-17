from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/draw_circle', methods=['POST'])
def draw_circle():
    radius = float(request.form.get("radius", 5))  # POSTリクエストで半径を取得

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

if __name__ == '__main__':
    app.run(debug=True)