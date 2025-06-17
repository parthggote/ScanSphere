# ScanSphere Node.js Backend

This is the Node.js backend for the ScanSphere application, built with Express.js and MySQL.

## Setup Instructions

### Prerequisites
- Node.js 14.x or higher
- MySQL server installed and running

### MySQL Setup
1. Make sure your MySQL server is installed and running
2. Create a MySQL user or use an existing one (like 'root')
3. Note the username, password, host, and port for the connection

### Installation

1. Install dependencies:
   ```
   cd node-backend
   npm install
   ```

2. Configure your MySQL database connection:
   - Copy the `.env.example` file to `.env` (if needed)
   - Edit the `.env` file with your MySQL credentials:
     ```
     DB_HOST=localhost
     DB_USER=root
     DB_PASSWORD=your_password
     DB_PORT=3306
     DB_NAME=scansphere
     ```
   - The default port for MySQL is usually 3306
   - The database `scansphere` will be created automatically if it doesn't exist

3. Test the MySQL connection:
   ```
   node test-connection.js
   ```
   This will verify that your MySQL credentials are correct and create the database if needed.

4. Start the server:
   ```
   npm start
   ```
   
   For development with auto-restart:
   ```
   npm run dev
   ```

The API will be available at http://localhost:5000

## Troubleshooting MySQL Connection

If you have issues connecting to MySQL:

1. Verify MySQL server is running:
   - Windows: Check Services or Task Manager
   - Linux: `sudo systemctl status mysql`
   - macOS: `brew services list | grep mysql`

2. Check your credentials in the `.env` file

3. Ensure the MySQL user has appropriate permissions:
   ```sql
   CREATE USER 'username'@'localhost' IDENTIFIED BY 'password';
   GRANT ALL PRIVILEGES ON scansphere.* TO 'username'@'localhost';
   FLUSH PRIVILEGES;
   ```

4. If using a remote MySQL server, ensure it's configured to accept remote connections and the port is open in your firewall.

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/register` - Register a new user
- `POST /api/login` - Log in a user
- `POST /api/forgot-password` - Request password reset

## Integrating with Frontend

Update your frontend JavaScript to make API calls to this backend. The forgot-password.html already includes the integration code.

Example for login form:

```javascript
document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    
    try {
        const response = await fetch('http://localhost:5000/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password }),
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Store the token for authenticated requests
            localStorage.setItem('token', data.token);
            // Redirect to dashboard or home page
            window.location.href = 'index.html';
        } else {
            // Show error message
            alert(data.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    }
});
``` 