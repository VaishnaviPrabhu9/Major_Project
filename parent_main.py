from flask import Flask, render_template, request, redirect, flash, session
from flask_mysqldb import MySQL
from MySQLdb.cursors import DictCursor
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'manas'

mysql = MySQL(app)

# Parent login route
@app.route('/parent_login', methods=['GET', 'POST'])
def parent_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Validate parent credentials
        cursor = mysql.connection.cursor(DictCursor)
        cursor.execute("SELECT * FROM parent WHERE email = %s AND password = %s", (email, password))
        parent = cursor.fetchone()
        cursor.close()

        if parent:
            session['parent_logged_in'] = True
            session['parent_id'] = parent['parent_registration']
            session['student_id'] = parent['student_id']
            flash("Login successful!", "success")
            return redirect('/parent_dashboard')
        else:
            flash("Invalid email or password!", "danger")
            return redirect('/parent_login')

    return render_template('parent_login.html')

# Parent dashboard to view student details
@app.route('/parent_dashboard', methods=['GET'])
def parent_dashboard():
    if 'parent_logged_in' not in session:
        flash("Please log in to access the dashboard.", "danger")
        return redirect('/parent_login')

    student_id = session.get('student_id')
    
    # Fetch student details based on student_id
    cursor = mysql.connection.cursor(DictCursor)
    cursor.execute("""
        SELECT student_registration, student_name, student_id, class, section, age, weight_kg, contact_number, created_at, FOMO, DASS, ADHD, feedback
        FROM student
        WHERE student_id = %s
    """, (student_id,))
    student = cursor.fetchone()
    cursor.close()

    if not student:
        flash("Student details not found.", "danger")
        return redirect('/parent_login')

    return render_template('parent_dashboard.html', student=student)

# Logout route
@app.route('/logout', methods=['GET'])
def logout():
    session.clear()
    flash("Logged out successfully!", "success")
    return redirect('/parent_login')

if __name__ == '__main__':
    app.run(debug=True)
