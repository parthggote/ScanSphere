const mysql = require('mysql2');
require('dotenv').config();
const fs = require('fs');
const { exec } = require('child_process');

// Get configuration from environment or use defaults
const config = {
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT) || 3306,
  user: process.env.DB_USER || 'root',
  password: process.env.DB_PASSWORD || '',
  database: process.env.DB_NAME || 'Scansphere',
  connectTimeout: 10000 // 10 seconds
};

console.log('\nSimple MySQL Connection Test');
console.log('===========================');
console.log('Current configuration:');
console.log(`Host: ${config.host}`);
console.log(`Port: ${config.port}`);
console.log(`User: ${config.user}`);
console.log(`Password: ${config.password ? '*'.repeat(config.password.length) : '<none>'}`);
console.log(`Database: ${config.database}`);
console.log('===========================');

// Detect MySQL service status
function checkMySQLService() {
  return new Promise((resolve) => {
    const isWindows = process.platform === 'win32';
    
    if (isWindows) {
      exec('sc query MySQL', (error, stdout) => {
        if (error) {
          console.log('Could not detect MySQL service status on Windows.');
          resolve(false);
          return;
        }
        
        if (stdout.includes('RUNNING')) {
          console.log('✅ MySQL service is running on Windows.');
          resolve(true);
        } else {
          console.log('❌ MySQL service is NOT running on Windows.');
          resolve(false);
        }
      });
    } else {
      // Linux/Mac
      exec('pgrep mysqld || pgrep mysql', (error) => {
        if (error) {
          console.log('❌ MySQL service does not appear to be running on Linux/Mac.');
          resolve(false);
          return;
        }
        
        console.log('✅ MySQL service is running on Linux/Mac.');
        resolve(true);
      });
    }
  });
}

// Check port availability
function checkPort() {
  return new Promise((resolve) => {
    const net = require('net');
    const socket = new net.Socket();
    
    socket.setTimeout(1000);
    
    socket.on('connect', () => {
      console.log(`✅ Port ${config.port} is open and accepting connections.`);
      socket.destroy();
      resolve(true);
    });
    
    socket.on('timeout', () => {
      console.log(`❌ Connection to port ${config.port} timed out. The port might be blocked or MySQL is not listening.`);
      socket.destroy();
      resolve(false);
    });
    
    socket.on('error', (error) => {
      console.log(`❌ Port ${config.port} error: ${error.message}`);
      resolve(false);
    });
    
    socket.connect(config.port, config.host);
  });
}

// Test MySQL connection
async function testConnection() {
  console.log('\nAttempting to connect to MySQL server...');
  
  // Create connection without specifying database first
  const tempConfig = { ...config };
  delete tempConfig.database;
  
  const connection = mysql.createConnection(tempConfig);
  
  return new Promise((resolve) => {
    connection.connect((err) => {
      if (err) {
        console.log(`Connection error: ${err.message}`);
        
        console.log('\nTroubleshooting tips:');
        if (err.code === 'ECONNREFUSED') {
          console.log('1. MySQL server is not running. Start the MySQL service.');
          console.log('2. The host or port might be incorrect.');
          console.log('3. Check if MySQL is running on a different port.');
        } else if (err.code === 'ER_ACCESS_DENIED_ERROR') {
          console.log('1. The username or password is incorrect.');
          console.log('2. Check your .env file for correct credentials.');
          console.log('3. Verify that this user has permissions to connect from this host.');
        } else {
          console.log('1. Check your MySQL configuration.');
          console.log('2. Verify your .env file has the correct settings.');
          console.log('3. Make sure MySQL is properly installed.');
        }
        
        try {
          connection.end();
        } catch (e) {
          // Ignore connection end errors
        }
        
        resolve(false);
        return;
      }
      
      console.log('✅ Successfully connected to MySQL server!');
      
      // Now check if database exists
      connection.query(`SHOW DATABASES LIKE '${config.database}'`, (err, results) => {
        if (err) {
          console.log(`Error checking database: ${err.message}`);
          connection.end();
          resolve(false);
          return;
        }
        
        if (results.length === 0) {
          console.log(`\n❌ Database '${config.database}' does not exist.`);
          console.log('Would you like to create it? (Y/n)');
          
          // Auto-create in non-interactive environments
          connection.query(`CREATE DATABASE \`${config.database}\``, (err) => {
            if (err) {
              console.log(`\nError creating database: ${err.message}`);
              console.log('You might not have permissions to create databases.');
            } else {
              console.log(`\n✅ Database '${config.database}' created successfully!`);
            }
            
            connection.end();
            resolve(true);
          });
          return;
        }
        
        console.log(`\n✅ Database '${config.database}' exists.`);
        
        // Now connect with the database and test queries
        connection.changeUser({ database: config.database }, (err) => {
          if (err) {
            console.log(`Error selecting database: ${err.message}`);
            connection.end();
            resolve(false);
            return;
          }
          
          console.log(`✅ Successfully connected to database '${config.database}'`);
          
          // Test query
          connection.query('SHOW TABLES', (err, results) => {
            if (err) {
              console.log(`Error running query: ${err.message}`);
            } else {
              console.log(`\nFound ${results.length} tables in database.`);
              
              if (results.length > 0) {
                console.log('Tables:');
                results.forEach(row => {
                  const tableName = row[`Tables_in_${config.database}`];
                  console.log(`- ${tableName}`);
                });
              }
            }
            
            // All done
            connection.end();
            resolve(true);
          });
        });
      });
    });
  });
}

// Generate a configuration file for MySQL Workbench
function generateConfigFile() {
  const configContent = `# MySQL Configuration
# Generated for easy troubleshooting
HOST=${config.host}
PORT=${config.port}
USER=${config.user}
PASSWORD=${config.password}
DATABASE=${config.database}
`;

  fs.writeFileSync('mysql-config.txt', configContent);
  console.log('\nMySQL configuration saved to mysql-config.txt');
}

// Main function
async function main() {
  try {
    await checkMySQLService();
    await checkPort();
    await testConnection();
    generateConfigFile();
    
    console.log('\nTest completed. Check the results above for any issues.');
    console.log('If MySQL is not running, you need to start it before running your application.');
    console.log('For Windows: Start MySQL service from Services or run "net start MySQL"');
    console.log('For Linux: Run "sudo systemctl start mysql" or "sudo service mysql start"');
    
    process.exit(0);
  } catch (error) {
    console.error('Unexpected error:', error);
    process.exit(1);
  }
}

// Run the test
main(); 