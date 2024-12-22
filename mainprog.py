from flask import Flask, render_template, request, redirect, flash, session

from flask_mysqldb import MySQL
from MySQLdb.cursors import DictCursor
import os
import pandas as pd
from werkzeug.security import generate_password_hash
app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'manas'

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/gohome')
def homepage():
    return render_template('index.html')

@app.route('/student')
def servicepage():
    return render_template('student_login.html')

@app.route('/admin')
def servicepage1():
    return render_template('admin_login.html')

@app.route('/parent')
def servicepage2():
    return render_template('parent_login.html')

@app.route('/parentreg')
def parreg():
    return render_template('parent_register.html')

@app.route('/adminreg')
def admreg():
    return render_template('admin_register.html')

@app.route('/studentreg')
def stureg():
    return render_template('student_register.html')

@app.route('/about')
def aboutpage():
    return render_template('about.html')

# Initialize MySQL
mysql = MySQL(app)

@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        student_name = request.form['student-name']
        student_id = request.form['student-id']
        class_name = request.form['class']
        section = request.form['section']
        age = request.form['age']
        weight = request.form['weight']
        contact = request.form['contact']
        password = request.form['password']
        confirm_password = request.form['confirm-password']

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect('/')

        # Database operation
        cur = mysql.connection.cursor()  # Corrected usage
        cur.execute('''
            INSERT INTO student (student_name, student_id, class, section, age, weight_kg, contact_number, password, confirm_password)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (student_name, student_id, class_name, section, age, weight, contact, password, confirm_password))
        mysql.connection.commit()
        cur.close()

        flash('Student registered successfully!', 'success')
        return redirect('/')


@app.route('/register_parent', methods=['GET', 'POST'])
def register_parent():
    if request.method == 'POST':
        # Get form data
        parent_name = request.form['parent_name']
        student_id = request.form['student_id']
        contact_number = request.form['contact_number']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect('/register_parent')

        # Check if the student is registered
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM student WHERE student_id = %s", (student_id,))
        student = cursor.fetchone()

        if not student:
            flash('Student ID not found. Register the student first.', 'danger')
            return redirect('/register_parent')

        # Insert parent data
        cursor.execute("""
            INSERT INTO parent (parent_name, student_id, contact_number, email, password)
            VALUES (%s, %s, %s, %s, %s)
        """, (parent_name, student_id, contact_number, email, password))
        mysql.connection.commit()
        cursor.close()

        flash('Parent registration successful!', 'success')
        return redirect('/login_parent')

    return render_template('parent_register.html')

        
@app.route('/userlogin')
def user_login():
   return render_template("login.html")
@app.route('/login_parent', methods=['GET', 'POST'])
def login_parent():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor = mysql.connection.cursor(DictCursor)
        cursor.execute("SELECT * FROM parent WHERE email = %s AND password = %s", (email, password))
        parent = cursor.fetchone()

        if parent:
            session['parent_id'] = parent['parent_registration']
            flash("Login successful!", "success")
            return redirect('/parent_dashboard')
        else:
            flash("Invalid email or password!", "danger")
            return redirect('/login_parent')

    return render_template('parent_login.html')
@app.route('/parent_dashboard', methods=['GET'])
def parent_dashboard():
    # Ensure the parent is logged in
    if 'parent_id' not in session:
        flash("Please log in to access the dashboard.", "warning")
        return redirect('/login_parent')
    
    # Retrieve the logged-in parent's ID
    parent_id = session['parent_id']
    
    cursor = mysql.connection.cursor(DictCursor)
    
    # Fetch parent details
    cursor.execute("SELECT * FROM parent WHERE parent_registration = %s", (parent_id,))
    parent = cursor.fetchone()
    
    # Fetch the associated student details
    cursor.execute("SELECT * FROM student WHERE student_id = %s", (parent['student_id'],))
    student = cursor.fetchone()
    
    cursor.close()
    
    return render_template('parent_dashboard.html', parent=parent, student=student)


# View Child Information
@app.route('/view_child', methods=['GET', 'POST'])
def view_child():
    print(request.method)  # Debugging: Print the request method
    if 'logged_in' in session:
        student_id = session['student_id']

        # Fetch student data
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM student WHERE student_id = %s", (student_id,))
        student = cursor.fetchone()

        if student:
            return render_template('view_child.html', student=student)

    flash('Please log in to view your child\'s information.', 'danger')
    return redirect('/login_parent')


# Logout
@app.route('/logout')
def logout():
    session.clear()
    flash('You have logged out successfully.', 'success')
    return redirect('/login_parent')




@app.route('/predictinfo')
def predictin():
   return render_template('info.html')









@app.route('/admin_dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    cursor = mysql.connection.cursor()

    # Fetch all student IDs for the dropdown
    if request.method == 'GET':
        cursor.execute("SELECT student_id, student_name FROM student")
        student_list = cursor.fetchall()
        cursor.close()
        return render_template('admin_dashboard.html', student_list=student_list)

    # Handle student selection
    if request.method == 'POST':
        selected_student_id = request.form.get('student_id')
        student_query = "SELECT * FROM student WHERE student_id = %s"
        cursor.execute(student_query, (selected_student_id,))
        student_details = cursor.fetchone()
        cursor.close()

        if student_details:
            return render_template('admin_dashboard.html', student_details=student_details, student_list=[])
        else:
            return "No details found for the selected student."






if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)

