from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
import os
from test import BreastCancerDetector
from test import test_image
from datetime import datetime, timedelta

app = Flask(__name__)

# Configure MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_PORT'] = 3306  
app.config['MYSQL_USER'] = 'root'  
app.config['MYSQL_PASSWORD'] = ' '  
app.config['MYSQL_DB'] = 'breast_cancer'  
mysql = MySQL(app)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

detector = BreastCancerDetector(
    benign_malignant_model_path='Models/model.h5',
    stage_model_path='Models/stage_classification_model.keras',
    svm_model_path='Models/stages_svm_model.pkl'
)

def allowed_file(filename):
    """Check if the file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        cursor = mysql.connection.cursor()

        # Check if user already exists
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user:
            flash('Email already exists!', 'danger')
            return redirect(url_for('register'))

        # Insert new user
        cursor.execute("INSERT INTO users (name, email, password, total_files_uploaded) VALUES (%s, %s, %s, %s)",
                       (name, email, password, 0))
        mysql.connection.commit()
        cursor.close()

        flash('Registration successful!', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if the user is an admin
        if email == "admin@gmail.com" and password == "admin@123":
            session['logged_in'] = True
            session['is_admin'] = True
            #flash('Admin Login successful!', 'success')
            return redirect(url_for('admin_dashboard'))

        # Regular user authentication
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user and user[3] == password:  # Check password
            session['logged_in'] = True
            session['user_id'] = user[0]
            session['name'] = user[1]
            session['email'] = user[2]

            # Get current login time
            login_time = datetime.now()

            # Update last login time & insert current login time
            cursor.execute("""
                UPDATE users 
                SET last_login_time = login_time, login_time = %s, files_uploaded = 0 
                WHERE id = %s
            """, (login_time, user[0]))
            mysql.connection.commit()

            cursor.close()
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor()
    cursor.execute("SELECT is_admin FROM users WHERE id = %s", (session['user_id'],))
    user_role = cursor.fetchone()
    cursor.close()

    if user_role and user_role[0] == 1:  # If the user is admin
        return render_template('admin_dashboard.html', name=session['name'], email=session['email'])

    return render_template('dashboard.html', name=session['name'], email=session['email'])


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'logged_in' not in session:
        return jsonify({'error': 'Unauthorized access'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result = detector.predict(filepath)
            result['image_path'] = url_for('static', filename=f'uploads/{filename}')
            
            # Update file upload count
            cursor = mysql.connection.cursor()
            cursor.execute("""
                UPDATE users 
                SET files_uploaded = files_uploaded + 1, total_files_uploaded = total_files_uploaded + 1 
                WHERE id = %s
            """, (session['user_id'],))
            mysql.connection.commit()
            cursor.close()

            os.remove(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/profile')
def profile():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM users WHERE id = %s", [session['user_id']])
    user = cursor.fetchone()
    cursor.close()
    print(user)
    return render_template('profile.html', user=user)



@app.route('/admin/dashboard')
def admin_dashboard():
    # Ensure only admin can access
    if 'logged_in' not in session or not session.get('is_admin'):
        flash("Unauthorized access!", "danger")
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor()

    # Get total users
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    # Get total files uploaded by all users
    cursor.execute("SELECT SUM(total_files_uploaded) FROM users")
    total_files_uploaded = cursor.fetchone()[0] or 0

    cursor.close()

    return render_template('admin_dashboard.html', total_users=total_users, total_files_uploaded=total_files_uploaded)

@app.route('/users')
def users():
    cursor = mysql.connection.cursor()

    query = "SELECT * FROM users"
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    #print(data)
    return render_template('users.html', users=data)

@app.route('/logout')
def logout():
    if 'logged_in' in session:
        user_id = session['user_id']
        cursor = mysql.connection.cursor()

        # Get logout time
        logout_time = datetime.now()

        # Get login time for duration calculation
        cursor.execute("SELECT login_time FROM users WHERE id = %s", (user_id,))
        login_time = cursor.fetchone()[0]

        if login_time:
            time_duration = logout_time - login_time
            time_duration_str = str(timedelta(seconds=time_duration.total_seconds()))  # Convert to HH:MM:SS

            # Update logout time and session duration
            cursor.execute("""
                UPDATE users 
                SET logout_time = %s, time_duration = %s 
                WHERE id = %s
            """, (logout_time, time_duration_str, user_id))
            mysql.connection.commit()

        cursor.close()
        session.clear()
        flash('You have been logged out!', 'success')

    return redirect(url_for('home'))


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
