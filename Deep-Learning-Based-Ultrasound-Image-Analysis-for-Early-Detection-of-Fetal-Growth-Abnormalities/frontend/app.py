import sys
import os
from flask import Flask, render_template

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("================================================================================")
    print(" FRONTEND SERVER STARTED")
    print("================================================================================")
    print(" Open browser at: http://localhost:3000")
    print(" Ensure backend is running separately on port 5000!")
    print("================================================================================")
    app.run(port=3000, debug=True)
