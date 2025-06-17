const mysql = require('mysql2/promise');
require('dotenv').config();

async function testConnection() {
  console.log('Testing MySQL connection...');
  console.log('Configuration:');
  console.log(`Host: ${process.env.DB_HOST}`);
  console.log(`Port: ${process.env.DB_PORT}`);
  console.log(`User: ${process.env.DB_USER}`);
  console.log(`Password: ${'*'.repeat(process.env.DB_PASSWORD ? process.env.DB_PASSWORD.length : 0)}`);
  console.log(`Database: ${process.env.DB_NAME}`);
  
  try {
    // First try to connect to MySQL server without specifying a database
    const tempConfig = {
      host: process.env.DB_HOST,
      port: process.env.DB_PORT,
      user: process.env.DB_USER,
      password: process.env.DB_PASSWORD
    };
    
    console.log('\nAttempting to connect to MySQL server...');
    const tempConnection = await mysql.createConnection(tempConfig);
    console.log('Connected to MySQL server successfully.');
    
    // Check if database exists
    const [rows] = await tempConnection.query(
      `SHOW DATABASES LIKE '${process.env.DB_NAME}'`
    );
    
    if (rows.length === 0) {
      console.log(`\nDatabase '${process.env.DB_NAME}' does not exist.`);
      console.log(`Creating database '${process.env.DB_NAME}'...`);
      
      await tempConnection.query(
        `CREATE DATABASE ${process.env.DB_NAME}`
      );
      
      console.log(`Database '${process.env.DB_NAME}' created successfully.`);
    } else {
      console.log(`\nDatabase '${process.env.DB_NAME}' already exists.`);
    }
    
    // Close temporary connection
    await tempConnection.end();
    
    // Now try to connect to the specific database
    const config = {
      host: process.env.DB_HOST,
      port: process.env.DB_PORT,
      user: process.env.DB_USER,
      password: process.env.DB_PASSWORD,
      database: process.env.DB_NAME
    };
    
    console.log(`\nAttempting to connect to database '${process.env.DB_NAME}'...`);
    const connection = await mysql.createConnection(config);
    console.log(`Connected to database '${process.env.DB_NAME}' successfully.`);
    
    // Close connection
    await connection.end();
    console.log('\nConnection test completed successfully.');
    
  } catch (error) {
    console.error('\nError connecting to MySQL database:');
    console.error(error.message);
    console.error('\nPossible solutions:');
    console.error('1. Check if MySQL server is running');
    console.error('2. Verify the credentials in the .env file');
    console.error('3. Make sure you have permission to create databases (if needed)');
    console.error('4. Check that the host and port are correct');
  }
}

// Run the test
testConnection(); 