from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///logs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class TrafficLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.String(100))
    count_in = db.Column(db.Integer)
    count_out = db.Column(db.Integer)
    camera_id = db.Column(db.String(50))
    posture = db.Column(db.String(100))
    alert = db.Column(db.String(200))


@app.route('/')
def index():
    camera_id = request.args.get('camera_id', None)
    start_time = request.args.get('start_time', None)
    end_time = request.args.get('end_time', None)

    query = TrafficLog.query

    if camera_id:
        query = query.filter(TrafficLog.camera_id == camera_id)

    if start_time and end_time:
        query = query.filter(TrafficLog.time.between(start_time, end_time))

    logs = query.all()

    return render_template('index.html', logs=logs)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()   # ✅ IMPORTANT

    app.run(debug=True)