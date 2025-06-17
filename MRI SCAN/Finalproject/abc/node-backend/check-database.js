const mysql = require('mysql2/promise');
require('dotenv').config();

async function checkDatabase() {
  try {
    console.log('Database Check Utility');
    console.log('=====================');
    console.log('Configuration:');
    console.log(`Host: ${process.env.DB_HOST}`);
    console.log(`Port: ${process.env.DB_PORT}`);
    console.log(`User: ${process.env.DB_USER}`);
    console.log(`Password: ${'*'.repeat(process.env.DB_PASSWORD ? process.env.DB_PASSWORD.length : 0)}`);
    console.log(`Database: ${process.env.DB_NAME}`);
    console.log('=====================\n');

    // First connect without database to check if it exists
    const tempConfig = {
      host: process.env.DB_HOST,
      user: process.env.DB_USER,
      password: process.env.DB_PASSWORD,
      port: parseInt(process.env.DB_PORT) || 3306
    };
    
    console.log('Connecting to MySQL server...');
    const tempConn = await mysql.createConnection(tempConfig);
    console.log('Connected to MySQL server successfully.');
    
    // Check if database exists
    const [dbResults] = await tempConn.query(`SHOW DATABASES LIKE '${process.env.DB_NAME}'`);
    
    if (dbResults.length === 0) {
      console.log(`\nDatabase '${process.env.DB_NAME}' does not exist.`);
      console.log('Creating database...');
      
      // Create the database
      await tempConn.query(`CREATE DATABASE \`${process.env.DB_NAME}\``);
      console.log(`Database '${process.env.DB_NAME}' created successfully.`);
    } else {
      console.log(`\nDatabase '${process.env.DB_NAME}' exists.`);
    }
    
    await tempConn.end();
    
    // Now connect with the database
    const config = {
      host: process.env.DB_HOST,
      user: process.env.DB_USER,
      password: process.env.DB_PASSWORD,
      database: process.env.DB_NAME,
      port: parseInt(process.env.DB_PORT) || 3306
    };
    
    console.log(`\nConnecting to database '${process.env.DB_NAME}'...`);
    const conn = await mysql.createConnection(config);
    console.log(`Connected to database '${process.env.DB_NAME}' successfully.`);
    
    // Check if users table exists
    const [tableResults] = await conn.query(`
      SELECT TABLE_NAME 
      FROM information_schema.TABLES 
      WHERE TABLE_SCHEMA = ? AND TABLE_NAME = 'users'
    `, [process.env.DB_NAME]);
    
    if (tableResults.length === 0) {
      console.log('\nUsers table does not exist.');
      console.log('Creating users table...');
      
      // Create the users table
      await conn.query(`
        CREATE TABLE users (
          id INT AUTO_INCREMENT PRIMARY KEY,
          username VARCHAR(80) NOT NULL UNIQUE,
          email VARCHAR(120) NOT NULL UNIQUE,
          password VARCHAR(255) NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
      `);
      console.log('Users table created successfully.');
    } else {
      console.log('\nUsers table exists.');
      
      // Check table structure
      const [columns] = await conn.query(`SHOW COLUMNS FROM users`);
      console.log('\nTable structure:');
      columns.forEach(col => {
        console.log(`- ${col.Field}: ${col.Type} ${col.Null === 'NO' ? 'NOT NULL' : ''} ${col.Key === 'PRI' ? 'PRIMARY KEY' : ''} ${col.Extra}`);
      });
      
      // Check if table has any records
      const [countResult] = await conn.query(`SELECT COUNT(*) as count FROM users`);
      console.log(`\nTotal records in users table: ${countResult[0].count}`);
      
      if (countResult[0].count > 0) {
        const [sampleUsers] = await conn.query(`SELECT id, username, email, created_at FROM users LIMIT 5`);
        console.log('\nSample users:');
        sampleUsers.forEach(user => {
          console.log(`- ID: ${user.id}, Username: ${user.username}, Email: ${user.email}, Created: ${user.created_at}`);
        });
      }
    }
    
    // Test inserting a record
    console.log('\nTesting data insertion...');
    try {
      const testUsername = `test_user_${Date.now()}`;
      const testEmail = `test_${Date.now()}@example.com`;
      
      await conn.query(`
        INSERT INTO users (username, email, password) 
        VALUES (?, ?, ?)
      `, [testUsername, testEmail, 'test_password']);
      
      console.log('Test record inserted successfully!');
      
      // Verify the insertion
      const [insertedUser] = await conn.query(`
        SELECT id, username, email FROM users WHERE email = ?
      `, [testEmail]);
      
      if (insertedUser.length > 0) {
        console.log(`Verified: User ${insertedUser[0].username} with ID ${insertedUser[0].id} was inserted.`);
      }
      
      // Clean up the test record
      await conn.query(`DELETE FROM users WHERE email = ?`, [testEmail]);
      console.log('Test record removed.');
    } catch (error) {
      console.error('Error during insertion test:', error.message);
      console.log('\nPossible issues:');
      console.log('1. Check table permissions');
      console.log('2. Verify column constraints are not being violated');
      console.log('3. Check for duplicate key issues');
    }
    
    await conn.end();
    console.log('\nDatabase check completed.');
    
  } catch (error) {
    console.error('\nError during database check:', error.message);
    if (error.code === 'ER_ACCESS_DENIED_ERROR') {
      console.log('\nAccess denied. Check your username and password.');
    } else if (error.code === 'ECONNREFUSED') {
      console.log('\nConnection refused. Make sure MySQL is running and the host/port are correct.');
    } else if (error.code === 'ER_BAD_DB_ERROR') {
      console.log(`\nDatabase '${process.env.DB_NAME}' does not exist or cannot be accessed.`);
    } else {
      console.log('\nCheck MySQL server status and configuration.');
    }
  }
}

// Run the check
checkDatabase(); 