require('dotenv').config();
const mysql = require('mysql2/promise');

async function main() {
  console.log('Starting direct database test...');
  console.log('Database config:');
  console.log('- Host:', process.env.DB_HOST);
  console.log('- User:', process.env.DB_USER);
  console.log('- DB:', process.env.DB_NAME);

  try {
    // Create a direct connection
    const connection = await mysql.createConnection({
      host: process.env.DB_HOST,
      user: process.env.DB_USER,
      password: process.env.DB_PASSWORD,
      database: process.env.DB_NAME
    });

    console.log('Connection established successfully');
    
    // Test query to check users table
    console.log('\nTesting users table:');
    const [tables] = await connection.query(`
      SHOW TABLES LIKE 'users'
    `);
    
    if (tables.length === 0) {
      console.log('❌ Users table not found');
    } else {
      console.log('✅ Users table exists');
      
      // Get table structure
      console.log('\nTable structure:');
      const [columns] = await connection.query('DESCRIBE users');
      columns.forEach(col => {
        console.log(`- ${col.Field}: ${col.Type}${col.Null === 'NO' ? ' NOT NULL' : ''}${col.Key === 'PRI' ? ' PRIMARY KEY' : ''}${col.Key === 'UNI' ? ' UNIQUE' : ''}`);
      });
      
      // Count users
      const [countResult] = await connection.query('SELECT COUNT(*) as count FROM users');
      console.log(`\nTotal users: ${countResult[0].count}`);
      
      // Get all users
      if (countResult[0].count > 0) {
        console.log('\nUser data:');
        const [users] = await connection.query('SELECT * FROM users');
        
        users.forEach(user => {
          console.log(`ID: ${user.id}`);
          console.log(`Username: ${user.username}`);
          console.log(`Email: ${user.email}`);
          console.log(`Password: ${user.password}`);
          console.log(`Created: ${user.created_at}`);
          console.log('-------------------');
        });
      }
    }
    
    await connection.end();
    console.log('Test completed successfully');
  } catch (error) {
    console.error('Error during test:', error);
  }
}

main(); 