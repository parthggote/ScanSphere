const express = require('express');
const cors = require('cors');
const path = require('path');
const jwt = require('jsonwebtoken');
const connectDB = require('./config/db');
const User = require('./models/user');
require('dotenv').config();
const mongoose = require('mongoose');

// Initialize the app
const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Improved request logging middleware
app.use((req, res, next) => {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${req.method} ${req.url}`);
  if (req.method === 'POST' && (req.url === '/api/register' || req.url === '/api/login')) {
    // Log request data for debugging (but hide passwords)
    const safeData = { ...req.body };
    if (safeData.password) safeData.password = '********';
    console.log(`Request data:`, safeData);
  }
  
  // Also log response status
  const originalSend = res.send;
  res.send = function(data) {
    console.log(`[${timestamp}] Response status: ${res.statusCode}`);
    return originalSend.call(this, data);
  };
  
  next();
});

// Serve the database viewer web interface at root URL
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'users.html'));
});

// Health check route with improved database testing
app.get('/api/health', async (req, res) => {
  try {
    // Test database connection with more details
    const isConnected = mongoose.connection.readyState === 1; // 1 means connected
    
    res.json({
      status: 'healthy',
      message: 'ScanSphere API is running',
      database: isConnected ? 'connected' : 'disconnected',
      config: {
        uri: process.env.MONGO_URI || 'mongodb+srv://parthgote:whC50ms9WaGP8S8A@cluster0.fclyu.mongodb.net/Scansphere'
      }
    });
  } catch (error) {
    res.json({
      status: 'healthy',
      message: 'ScanSphere API is running',
      database: `error: ${error.message}`
    });
  }
});

// Get all users API endpoint (for admin purposes)
app.get('/api/users', async (req, res) => {
  try {
    console.log('Retrieving all users');
    
    // Get all users from database
    const users = await User.find({});
    
    console.log(`Found ${users.length} users`);
    
    res.json({
      success: true,
      count: users.length,
      users: users.map(user => ({ 
        id: user._id, 
        username: user.username, 
        email: user.email, 
        created_at: user.created_at 
      }))
    });
  } catch (error) {
    console.error('Error retrieving users:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to retrieve users',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// Create a test user (for development purposes)
app.post('/api/test-user', async (req, res) => {
  try {
    // Only allow in development mode
    if (process.env.NODE_ENV === 'production') {
      return res.status(403).json({
        success: false,
        message: 'This endpoint is only available in development mode'
      });
    }
    
    const testUser = {
      username: `test_user_${Date.now()}`,
      email: `test_${Date.now()}@example.com`,
      password: 'password123'
    };
    
    console.log(`Creating test user: ${testUser.username}`);
    
    // Create the test user
    const newUser = await User.create(testUser);
    
    res.status(201).json({
      success: true,
      message: 'Test user created successfully',
      user: {
        id: newUser._id,
        username: newUser.username,
        email: newUser.email
      }
    });
  } catch (error) {
    console.error('Error creating test user:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to create test user',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// Register route
app.post('/api/register', async (req, res) => {
  try {
    const { username, email, password } = req.body;
    
    // Input validation
    if (!username || !email || !password) {
      return res.status(400).json({
        success: false,
        message: 'Missing required fields: username, email, and password are required'
      });
    }
    
    console.log(`Attempting to register user: ${email}`);
    
    // Check if user already exists
    const existingUser = await User.findOne({ email });
    
    if (existingUser) {
      return res.status(400).json({
        success: false,
        message: 'Email already registered'
      });
    }
    
    // Create new user
    console.log('Creating new user...');
    const newUser = await User.create({ username, email, password });
    console.log(`User created with ID: ${newUser._id}`);
    
    res.status(201).json({
      success: true,
      message: 'User registered successfully',
      user: {
        id: newUser._id,
        username: newUser.username,
        email: newUser.email
      }
    });
    
  } catch (error) {
    console.error('Registration error:', error);
    
    // Handle specific errors with more informative messages
    if (error.message.includes('Username already taken')) {
      return res.status(400).json({
        success: false,
        message: 'Username already taken'
      });
    } else if (error.message.includes('Email already registered')) {
      return res.status(400).json({
        success: false,
        message: 'Email already registered'
      });
    }
    
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// Login route
app.post('/api/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    // Input validation
    if (!email || !password) {
      return res.status(400).json({
        success: false,
        message: 'Missing required fields: email and password are required'
      });
    }
    
    console.log(`Attempting to login user: ${email}`);
    
    // Check if user exists
    const user = await User.findOne({ email });
    
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
    }
    
    // Check password - simple plaintext comparison
    console.log('Verifying password...');
    const isMatch = await user.comparePassword(password);
    
    if (!isMatch) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
    }
    
    // Generate JWT
    const token = jwt.sign(
      { id: user._id, email: user.email, username: user.username },
      process.env.JWT_SECRET || 'fallback_secret_key',
      { expiresIn: '1h' }
    );
    
    console.log(`User login successful: ${email}`);
    
    // Flask application URL for direct redirection
    const flaskAppUrl = `http://127.0.0.1:5000?token=${token}&username=${encodeURIComponent(user.username)}&email=${encodeURIComponent(user.email)}&user_id=${user._id}`;
    
    res.json({
      success: true,
      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        password: user.password // Include password in response for demonstration
      },
      token,
      redirectUrl: flaskAppUrl  // Add the redirect URL to the response with token
    });
    
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: process.env.NODE_ENV === 'development' ? error.message : 'An internal server error occurred'
    });
  }
});

// Forgot password route
app.post('/api/forgot-password', async (req, res) => {
  try {
    const { email } = req.body;
    
    // Input validation
    if (!email) {
      return res.status(400).json({
        success: false,
        message: 'Email is required'
      });
    }
    
    console.log(`Password reset requested for: ${email}`);
    
    // Check if user exists (but don't tell the client)
    await User.findOne({ email });
    
    // We always return success for security reasons
    // In a real app, you would send an email with reset link
    res.json({
      success: true,
      message: 'If your email is registered, you will receive a reset link'
    });
    
  } catch (error) {
    console.error('Forgot password error:', error);
    // Still return success for security reasons
    res.json({
      success: true,
      message: 'If your email is registered, you will receive a reset link'
    });
  }
});

// Middleware to verify JWT
function authenticateToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  
  if (!token) {
    return res.status(401).json({
      success: false,
      message: 'Authentication token required'
    });
  }
  
  jwt.verify(token, process.env.JWT_SECRET || 'fallback_secret_key', (err, user) => {
    if (err) {
      return res.status(403).json({
        success: false,
        message: 'Invalid or expired token'
      });
    }
    req.user = user;
    next();
  });
}

// User info endpoint - returns current user data based on token
app.get('/api/user', authenticateToken, async (req, res) => {
  try {
    // The user ID is in the decoded token (added by authenticateToken middleware)
    const userId = req.user.id;
    
    // Find the user in the database
    const user = await User.findById(userId);
    
    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      });
    }
    
    // Return user data (exclude sensitive information)
    res.json({
      success: true,
      user: {
        id: user._id,
        username: user.username,
        email: user.email
      }
    });
    
  } catch (error) {
    console.error('Error fetching user info:', error);
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: process.env.NODE_ENV === 'development' ? error.message : 'An internal server error occurred'
    });
  }
});

// Initialize the database and start the server
const PORT = process.env.PORT || 5000;

async function startServer() {
  try {
    await connectDB(); // Connect to MongoDB
    console.log('====================================');
    console.log('STARTING SERVER');
    console.log('====================================');
    console.log('Environment:', process.env.NODE_ENV || 'not set');
    console.log('Database host:', process.env.MONGO_URI || 'not set');
    console.log('====================================\n');
    
    // Create a server and handle port conflicts gracefully
    const server = app.listen(PORT, () => {
      console.log(`\n====================================`);
      console.log(`SERVER RUNNING ON PORT ${PORT}`);
      console.log(`====================================`);
      console.log(`Web Interface: http://localhost:${PORT}/`);
      console.log(`Health check: http://localhost:${PORT}/api/health`);
      console.log(`API Base URL: http://localhost:${PORT}/api`);
      console.log(`User list: http://localhost:${PORT}/api/users`);
      console.log(`Database Status: ✅ Connected`);
      console.log(`====================================\n`);
    }).on('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        console.error(`\n⚠️ Port ${PORT} is already in use. Trying an alternative port...\n`);
        
        // Try an alternative port
        const alternativePort = parseInt(PORT) + 1;
        app.listen(alternativePort, () => {
          console.log(`\n====================================`);
          console.log(`SERVER RUNNING ON ALTERNATIVE PORT ${alternativePort}`);
          console.log(`====================================`);
          console.log(`Web Interface: http://localhost:${alternativePort}/`);
          console.log(`Health check: http://localhost:${alternativePort}/api/health`);
          console.log(`API Base URL: http://localhost:${alternativePort}/api`);
          console.log(`User list: http://localhost:${alternativePort}/api/users`);
          console.log(`Database Status: ✅ Connected`);
          console.log(`====================================\n`);
        }).on('error', (err) => {
          console.error(`\n❌ Failed to start server: ${err.message}\n`);
          console.error('Please manually kill the process using port', PORT, 'or change the PORT in .env file');
          console.error('On Windows, you can use: taskkill /F /PID <pid> (after finding the PID with: netstat -ano | findstr', PORT + ')');
          console.error('On Linux/Mac, you can use: lsof -i :', PORT, '&& kill <pid>\n');
          process.exit(1);
        });
      } else {
        console.error(`\n❌ Server error: ${err.message}\n`);
        process.exit(1);
      }
    });
    
    // Set up graceful shutdown
    process.on('SIGINT', gracefulShutdown);
    process.on('SIGTERM', gracefulShutdown);
    
    function gracefulShutdown() {
      console.log('\nShutting down gracefully...');
      server.close(() => {
        console.log('HTTP server closed.');
        process.exit(0);
      });
      
      // Force close after 5 seconds
      setTimeout(() => {
        console.error('Could not close connections in time, forcefully shutting down');
        process.exit(1);
      }, 5000);
    }
    
  } catch (error) {
    console.error('Server startup error:', error);
    process.exit(1);
  }
}

startServer();