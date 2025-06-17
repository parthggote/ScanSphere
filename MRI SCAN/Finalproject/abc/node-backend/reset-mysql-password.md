# MySQL Password Reset Guide

If you're getting the "Access denied for user 'root'@'localhost'" error, here are steps to fix it:

## Option 1: Try common default passwords

In your `.env` file, try these common default passwords one by one:
- Empty password: `DB_PASSWORD=`
- "root": `DB_PASSWORD=root`
- "password": `DB_PASSWORD=password`
- "mysql": `DB_PASSWORD=mysql`

## Option 2: Create a new MySQL user (Windows)

1. Open Command Prompt as Administrator

2. Stop the MySQL service:
   ```
   net stop mysql
   ```
   (If using MySQL as a Windows service, you might need to use `net stop "mysql80"` or similar)

3. Start MySQL in safe mode (without password):
   ```
   "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysqld.exe" --defaults-file="C:\ProgramData\MySQL\MySQL Server 8.0\my.ini" --init-file=C:\mysql-init.txt --console
   ```
   (Adjust the path to match your MySQL installation)

4. Create a `C:\mysql-init.txt` file with these contents:
   ```
   ALTER USER 'root'@'localhost' IDENTIFIED BY 'new_password';
   ```
   (Replace 'new_password' with your desired password)

5. After MySQL starts in safe mode and processes the init file, press Ctrl+C to stop it

6. Restart the MySQL service normally:
   ```
   net start mysql
   ```

7. Update your `.env` file with the new password:
   ```
   DB_PASSWORD=new_password
   ```

## Option 3: Create a new MySQL user (Alternative method)

If you can log in to MySQL with any user:

1. Open Command Prompt and login to MySQL:
   ```
   mysql -u username -p
   ```
   (Replace 'username' with any user that can log in)

2. Once logged in, create a new user with full privileges:
   ```sql
   CREATE USER 'scanuser'@'localhost' IDENTIFIED BY 'scanpassword';
   GRANT ALL PRIVILEGES ON *.* TO 'scanuser'@'localhost' WITH GRANT OPTION;
   FLUSH PRIVILEGES;
   exit;
   ```

3. Update your `.env` file to use this new user:
   ```
   DB_USER=scanuser
   DB_PASSWORD=scanpassword
   ```

## Option 4: Using MySQL Workbench

If you have MySQL Workbench installed:

1. Open MySQL Workbench
2. Connect to your MySQL server using any available method
3. Go to Server > Users and Privileges
4. Create a new account or reset the password for 'root'@'localhost'
5. Apply the changes
6. Update your `.env` file with the correct password

## Testing the Connection

After making changes, run the test connection script:
```
node test-connection.js
``` 