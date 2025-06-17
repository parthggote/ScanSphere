const mysql = require('mysql2');
require('dotenv').config();

// Connect to the database
const connection = mysql.createConnection({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  port: process.env.DB_PORT || 3306
});

console.log('====================================');
console.log('  MYSQL USER DATA VIEWER & MANAGER  ');
console.log('====================================');
console.log(`Database: ${process.env.DB_NAME}`);
console.log(`Table: users`);
console.log('====================================\n');

// Connect to the database
console.log('Connecting to MySQL...');
connection.connect((err) => {
  if (err) {
    console.error('❌ Connection error:', err.message);
    process.exit(1);
  }
  
  console.log('✅ Connected to MySQL database\n');
  showOptions();
});

// Show available options
function showOptions() {
  console.log('What would you like to do?');
  console.log('1. View all users');
  console.log('2. Search for a user');
  console.log('3. Add a test user');
  console.log('4. Delete a user');
  console.log('5. Exit');
  
  // Using Node.js readline for command-line input
  const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
  });
  
  readline.question('\nEnter your choice (1-5): ', (choice) => {
    readline.close();
    
    switch(choice) {
      case '1':
        viewAllUsers();
        break;
      case '2':
        searchUser();
        break;
      case '3':
        addTestUser();
        break;
      case '4':
        deleteUser();
        break;
      case '5':
        console.log('Goodbye!');
        connection.end();
        break;
      default:
        console.log('Invalid choice. Please try again.');
        showOptions();
    }
  });
}

// View all users
function viewAllUsers() {
  console.log('\n===== ALL REGISTERED USERS =====\n');
  
  connection.query(
    'SELECT id, username, email, DATE_FORMAT(created_at, "%Y-%m-%d %H:%i:%s") as created_at FROM users ORDER BY id DESC', 
    (err, results) => {
      if (err) {
        console.error('❌ Error fetching users:', err.message);
        returnToMenu();
        return;
      }
      
      if (results.length === 0) {
        console.log('No users found in the database.\n');
      } else {
        console.log(`Found ${results.length} users:\n`);
        
        // Display as a formatted table
        console.log('ID  | Username            | Email                           | Created At');
        console.log('----+---------------------+---------------------------------+------------------------');
        
        results.forEach(user => {
          const id = user.id.toString().padEnd(3);
          const username = user.username.padEnd(20);
          const email = user.email.padEnd(30);
          console.log(`${id} | ${username} | ${email} | ${user.created_at}`);
        });
        console.log('\n');
      }
      
      returnToMenu();
    }
  );
}

// Search for a user
function searchUser() {
  const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
  });
  
  readline.question('\nEnter search term (username or email): ', (searchTerm) => {
    readline.close();
    
    console.log(`\n===== SEARCH RESULTS FOR "${searchTerm}" =====\n`);
    
    connection.query(
      'SELECT id, username, email, DATE_FORMAT(created_at, "%Y-%m-%d %H:%i:%s") as created_at FROM users WHERE username LIKE ? OR email LIKE ?',
      [`%${searchTerm}%`, `%${searchTerm}%`],
      (err, results) => {
        if (err) {
          console.error('❌ Error searching users:', err.message);
          returnToMenu();
          return;
        }
        
        if (results.length === 0) {
          console.log('No matching users found.\n');
        } else {
          console.log(`Found ${results.length} matching users:\n`);
          
          // Display as a formatted table
          console.log('ID  | Username            | Email                           | Created At');
          console.log('----+---------------------+---------------------------------+------------------------');
          
          results.forEach(user => {
            const id = user.id.toString().padEnd(3);
            const username = user.username.padEnd(20);
            const email = user.email.padEnd(30);
            console.log(`${id} | ${username} | ${email} | ${user.created_at}`);
          });
          console.log('\n');
        }
        
        returnToMenu();
      }
    );
  });
}

// Add a test user
function addTestUser() {
  const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
  });
  
  readline.question('\nEnter username for test user: ', (username) => {
    readline.question('Enter email for test user: ', (email) => {
      readline.close();
      
      const testUser = {
        username: username || `test_user_${Date.now()}`,
        email: email || `test_${Date.now()}@example.com`,
        password: '$2a$10$eCJQvOQUD.VekJb8.9S5/.Y1RhxBRhxKmzI5jjRMXFZFBypJQgFEG' // Hashed 'password123'
      };
      
      console.log('\nAdding test user:');
      console.log(`Username: ${testUser.username}`);
      console.log(`Email: ${testUser.email}`);
      
      connection.query(
        'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
        [testUser.username, testUser.email, testUser.password],
        (err, result) => {
          if (err) {
            console.error('❌ Error adding test user:', err.message);
            returnToMenu();
            return;
          }
          
          console.log(`\n✅ Test user added successfully with ID: ${result.insertId}\n`);
          returnToMenu();
        }
      );
    });
  });
}

// Delete a user
function deleteUser() {
  const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
  });
  
  readline.question('\nEnter user ID to delete: ', (userId) => {
    readline.close();
    
    if (!userId) {
      console.log('No user ID provided.');
      returnToMenu();
      return;
    }
    
    // First check if the user exists
    connection.query(
      'SELECT username, email FROM users WHERE id = ?',
      [userId],
      (err, results) => {
        if (err) {
          console.error('❌ Error checking user:', err.message);
          returnToMenu();
          return;
        }
        
        if (results.length === 0) {
          console.log(`\nNo user found with ID: ${userId}\n`);
          returnToMenu();
          return;
        }
        
        const user = results[0];
        console.log(`\nAre you sure you want to delete this user?`);
        console.log(`ID: ${userId}`);
        console.log(`Username: ${user.username}`);
        console.log(`Email: ${user.email}`);
        
        const confirmRL = require('readline').createInterface({
          input: process.stdin,
          output: process.stdout
        });
        
        confirmRL.question('\nType "yes" to confirm: ', (confirmation) => {
          confirmRL.close();
          
          if (confirmation.toLowerCase() !== 'yes') {
            console.log('Deletion cancelled.');
            returnToMenu();
            return;
          }
          
          connection.query(
            'DELETE FROM users WHERE id = ?',
            [userId],
            (err, result) => {
              if (err) {
                console.error('❌ Error deleting user:', err.message);
                returnToMenu();
                return;
              }
              
              if (result.affectedRows === 0) {
                console.log(`\nNo user was deleted. User ID ${userId} not found.\n`);
              } else {
                console.log(`\n✅ User with ID ${userId} deleted successfully.\n`);
              }
              
              returnToMenu();
            }
          );
        });
      }
    );
  });
}

// Return to the main menu
function returnToMenu() {
  const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
  });
  
  readline.question('\nPress Enter to return to the menu...', () => {
    readline.close();
    console.log('\n');
    showOptions();
  });
} 