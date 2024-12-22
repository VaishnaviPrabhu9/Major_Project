from flask import Flask, request, jsonify, render_template
import mysql.connector

app = Flask(__name__)

# Database connection details
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",  # Replace with your database password
    "database": "manas"
}

# Route to serve the HTML dashboard
@app.route("/")
def dashboard():
    return render_template("parent_dashboard.html")

# Route to fetch student data
@app.route("/fetch_student_data", methods=["POST"])
def fetch_student_data():
    try:
        # Parse the student_id from the request
        data = request.get_json()
        student_id = data.get("student_id")

        if not student_id:
            return jsonify({"success": False, "message": "Student ID is required."})

        # Connect to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # Query to fetch student data
        query = "SELECT FOMO, DASS, ADHD FROM student WHERE student_id = %s"
        cursor.execute(query, (student_id,))
        student = cursor.fetchone()

        # Close the database connection
        cursor.close()
        conn.close()

        if student:
            return jsonify({"success": True, "student": student})
        else:
            return jsonify({"success": False, "message": "No data found for the provided Student ID."})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "message": "An error occurred. Please try again later."})

if __name__ == "__main__":
    app.run(debug=True)
