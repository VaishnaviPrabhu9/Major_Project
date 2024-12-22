from flask import Flask, render_template, request, redirect, flash, session
from flask_mysqldb import MySQL
from MySQLdb.cursors import DictCursor
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'manas'

mysql = MySQL(app)

# Admin login route
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == 'admin' and password == 'admin123':  # Hardcoded credentials
            session['admin_logged_in'] = True
            flash('Admin login successful!', 'success')
            return redirect('/admin_dashboard')
        else:
            flash('Invalid credentials!', 'danger')
            return redirect('/admin_login')

    return render_template('admin_login.html')

# Admin dashboard with student lookup
@app.route('/admin_dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if 'admin_logged_in' not in session:
        flash("Please log in as admin to access the dashboard.", "danger")
        return redirect('/admin_login')
    
    student = None

    if request.method == 'POST':
        student_id = request.form['student_id']
        cursor = mysql.connection.cursor(DictCursor)
        cursor.execute("SELECT * FROM student WHERE student_id = %s", (student_id,))
        student = cursor.fetchone()
        cursor.close()

        if not student:
            flash(f"No student found with ID {student_id}", "danger")

    return render_template('admin_dashboard.html', student=student)

# View all students in a table format
@app.route('/view_all_students', methods=['GET'])
def view_all_students():
    if 'admin_logged_in' not in session:
        flash("Please log in as admin to access this page.", "danger")
        return redirect('/admin_login')

    cursor = mysql.connection.cursor(DictCursor)
    cursor.execute("SELECT student_registration, student_name, student_id, class, section, age, weight_kg, contact_number, FOMO, DASS, ADHD, feedback FROM student")
    students = cursor.fetchall()
    cursor.close()

    return render_template('view_all_students.html', students=students)

# Update student feedback
@app.route('/update_feedback', methods=['GET', 'POST'])
def update_feedback():
    if 'admin_logged_in' not in session:
        flash("Please log in as admin to access this page.", "danger")
        return redirect('/admin_login')
    
    if request.method == 'POST':
        student_id = request.form['student_id']
        feedback = request.form['feedback']
        
        cursor = mysql.connection.cursor()
        cursor.execute("UPDATE student SET feedback = %s WHERE student_id = %s", (feedback, student_id))
        mysql.connection.commit()
        cursor.close()

        flash('Feedback updated successfully!', 'success')
        return redirect('/admin_dashboard')

    return render_template('update_feedback.html')

# View student information
@app.route('/view_student', methods=['GET', 'POST'])
def view_student():
    if request.method == 'POST':
        student_id = request.form['student_id']  # Get the student ID from the form

        cursor = mysql.connection.cursor(DictCursor)
        cursor.execute("""
            SELECT student_registration, student_name, student_id, class, section, age, weight_kg, contact_number, password, confirm_password, created_at, FOMO, DASS, ADHD, feedback
            FROM student
            WHERE student_id = %s
        """, (student_id,))
        student = cursor.fetchone()  # Fetch the student details

        cursor.close()

        if student:
            return render_template('student_details.html', student=student)
        else:
            flash("Student not found!", "danger")
            return redirect('/admin_dashboard')  # Redirect to admin dashboard if student not found

    return render_template('view_student.html')

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)
