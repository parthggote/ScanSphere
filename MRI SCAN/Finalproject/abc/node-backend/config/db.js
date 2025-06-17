const mongoose = require('mongoose');
require('dotenv').config();

const mongoURI = process.env.MONGO_URI || 'mongodb+srv://parthgote:whC50ms9WaGP8S8A@cluster0.fclyu.mongodb.net/Scansphere';

const connectDB = async () => {
  try {
    await mongoose.connect(mongoURI);
    console.log('MongoDB Connected successfully');
  } catch (err) {
    console.error('MongoDB Connection Failed:', err.message);
    // Exit process with failure
    process.exit(1);
  }
};

module.exports = connectDB; 