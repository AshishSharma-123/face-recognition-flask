from flask import Flask, render_template, request, session, jsonify
from face_recognition import generate_dataset
from face_recognition import train_classifier  # Import your training function
from face_recognition import run_recognition

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for sessions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm-password']
        if password == confirm_password:
            session['username'] = username
            return render_template('dashboard.html', username=username)
        return "Passwords do not match!"
    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username and password:
            session['username'] = username
            return render_template('dashboard.html', username=username)
        return "Invalid credentials!"
    return render_template('signin.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/create-dataset', methods=['POST'])
def create_dataset():
    data = request.get_json()
    name = data.get('name')
    if not name:
        return jsonify({'message': 'Dataset name is required!'}), 400

    try:
        count = generate_dataset(name)
        return jsonify({'message': f"Dataset '{name}' created with {count} images!"})
    except Exception as e:
        return jsonify({'message': f"Error: {str(e)}"}), 500

# ✅ New route for Train Model button
@app.route('/train-model')
def train_model():
    username = session.get('username', 'User')  # Get username from session
    try:
        label_map = train_classifier("data")
        return render_template('dashboard.html', username=username, message="Model trained successfully!")
    except Exception as e:
        return render_template('dashboard.html', username=username, message=f"Training failed: {str(e)}")
    
@app.route('/recognize-face')
def recognize_face():
    try:
        run_recognition()
        return render_template('dashboard.html', message="Face recognition completed!")
    except Exception as e:
        return render_template('dashboard.html', message=f"Recognition failed: {str(e)}")



if __name__ == '__main__':
    app.run(debug=True)
