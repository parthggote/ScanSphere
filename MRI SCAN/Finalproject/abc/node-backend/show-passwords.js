// Simple script to display passwords from the database
const { pool } = require('./config/db');

async function showPasswords() {
  try {
    console.log('Fetching users and passwords...');
    
    // Direct query to get all user data
    const [users] = await pool.query('SELECT * FROM users');
    
    console.log('\n==== USERS AND PASSWORDS ====');
    users.forEach(user => {
      console.log(`ID: ${user.id}`);
      console.log(`Email: ${user.email}`);
      console.log(`Password: ${user.password}`);
      console.log('--------------------------');
    });
    
    console.log(`\nTotal users: ${users.length}`);
    console.log('\nNote: These are bcrypt hashed passwords, not plaintext.');
    
  } catch (error) {
    console.error('Error:', error);
  } finally {
    process.exit();
  }
}

showPasswords(); 