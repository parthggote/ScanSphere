document.addEventListener('DOMContentLoaded', function() {
    // Set API URL based on current location
    const API_URL = window.location.origin + '/api';
    
    // Set the current year in the footer
    const currentYearElements = document.querySelectorAll('#current-year');
    currentYearElements.forEach(element => {
        element.textContent = new Date().getFullYear();
    });

    // Theme toggle functionality
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeToggleIcon(savedTheme);

    const themeToggleButtons = document.querySelectorAll('#theme-toggle');
    themeToggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeToggleIcon(newTheme);
        });
    });

    // Password toggle functionality
    const passwordToggles = document.querySelectorAll('.password-toggle');
    passwordToggles.forEach(toggle => {
        toggle.addEventListener('click', function() {
            const input = this.parentElement.querySelector('input');
            const type = input.getAttribute('type') === 'password' ? 'text' : 'password';
            input.setAttribute('type', type);
            
            // Update icon
            const icon = this.querySelector('i');
            if (type === 'text') {
                icon.classList.remove('fa-eye');
                icon.classList.add('fa-eye-slash');
            } else {
                icon.classList.remove('fa-eye-slash');
                icon.classList.add('fa-eye');
            }
        });
    });

    // Login form submission
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const submitButton = this.querySelector('button[type="submit"]');
            submitButton.classList.add('loading');
            
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            
            try {
                const response = await fetch(`${API_URL}/login`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ email, password })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    // Login successful
                    localStorage.setItem('user', JSON.stringify(data.user));
                    localStorage.setItem('token', data.token);
                    
                    // Redirect to Flask application
                    if (data.redirectUrl) {
                        console.log(`Redirecting to Flask application: ${data.redirectUrl}`);
                        
                        // Store token and user data in sessionStorage for the Flask app
                        sessionStorage.setItem('authToken', data.token);
                        sessionStorage.setItem('userData', JSON.stringify({
                            id: data.user.id,
                            username: data.user.username,
                            email: data.user.email
                        }));
                        
                        // Redirect the user to the Flask app
                        window.location.href = data.redirectUrl;
                    } else {
                        // Fallback to dashboard if no redirect URL provided
                        window.location.href = 'index.html';
                    }
                } else {
                    // Login failed
                    alert(`Login failed: ${data.message || 'Unknown error'}`);
                }
            } catch (error) {
                alert(`Error connecting to server: ${error.message}`);
                console.error('Login error:', error);
            } finally {
                submitButton.classList.remove('loading');
            }
        });
    }

    // Signup form submission
    const signupForm = document.getElementById('signup-form');
    if (signupForm) {
        const passwordInput = document.getElementById('signup-password');
        const confirmPasswordInput = document.getElementById('confirm-password');
        const passwordError = document.getElementById('password-error');
        const strengthSegments = document.querySelectorAll('.strength-segment');
        
        // Password strength meter
        if (passwordInput) {
            passwordInput.addEventListener('input', function() {
                updatePasswordStrength(this.value);
            });
        }
        
        // Password validation
        function validatePasswords() {
            if (passwordInput.value !== confirmPasswordInput.value) {
                passwordError.textContent = 'Passwords do not match';
                return false;
            }
            if (passwordInput.value.length < 8) {
                passwordError.textContent = 'Password must be at least 8 characters';
                return false;
            }
            passwordError.textContent = '';
            return true;
        }
        
        // Password strength meter
        function updatePasswordStrength(password) {
            let strength = 0;
            
            if (password.length >= 8) strength++;
            if (password.match(/[A-Z]/)) strength++;
            if (password.match(/[0-9]/)) strength++;
            if (password.match(/[^A-Za-z0-9]/)) strength++;
            
            strengthSegments.forEach((segment, index) => {
                segment.classList.remove('weak', 'medium', 'strong');
                
                if (index < strength) {
                    if (strength === 1) segment.classList.add('weak');
                    else if (strength === 2 || strength === 3) segment.classList.add('medium');
                    else segment.classList.add('strong');
                }
            });
        }
        
        signupForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            if (!validatePasswords()) {
                return;
            }
            
            const submitButton = this.querySelector('button[type="submit"]');
            submitButton.classList.add('loading');
            
            const username = document.getElementById('name').value;
            const email = document.getElementById('signup-email').value;
            const password = document.getElementById('signup-password').value;
            
            try {
                const response = await fetch(`${API_URL}/register`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ username, email, password })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    // Registration successful
                    alert('Registration successful! Please log in.');
                    window.location.href = 'login.html';
                } else {
                    // Registration failed
                    alert(`Registration failed: ${data.message || 'Unknown error'}`);
                }
            } catch (error) {
                alert(`Error connecting to server: ${error.message}`);
                console.error('Registration error:', error);
            } finally {
                submitButton.classList.remove('loading');
            }
        });
    }

    // Forgot password form
    const forgotPasswordForm = document.getElementById('forgot-password-form');
    if (forgotPasswordForm) {
        forgotPasswordForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const emailInput = document.getElementById('reset-email');
            const submitButton = this.querySelector('button[type="submit"]');
            submitButton.classList.add('loading');
            
            // Simulate password reset process
            setTimeout(() => {
                submitButton.classList.remove('loading');
                
                // Show success message
                document.getElementById('reset-form-container').classList.add('hidden');
                document.getElementById('reset-success').classList.remove('hidden');
                document.getElementById('sent-email').textContent = emailInput.value;
            }, 1500);
        });
        
        // Try again button
        const tryAgainButton = document.getElementById('try-again');
        if (tryAgainButton) {
            tryAgainButton.addEventListener('click', function() {
                document.getElementById('reset-success').classList.add('hidden');
                document.getElementById('reset-form-container').classList.remove('hidden');
            });
        }
    }

    // Animate stats counter
    const statValues = document.querySelectorAll('.stat-value');
    if (statValues.length > 0) {
        const options = {
            threshold: 0.5
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const target = entry.target;
                    const countTo = parseInt(target.getAttribute('data-count'));
                    
                    let count = 0;
                    const updateCount = () => {
                        const increment = countTo / 50;
                        if (count < countTo) {
                            count += increment;
                            target.textContent = Math.ceil(count);
                            requestAnimationFrame(updateCount);
                        } else {
                            target.textContent = countTo;
                        }
                    };
                    
                    updateCount();
                    observer.unobserve(target);
                }
            });
        }, options);
        
        statValues.forEach(value => {
            observer.observe(value);
        });
    }
});

// Helper function to update theme toggle icon
function updateThemeToggleIcon(theme) {
    const themeToggleButtons = document.querySelectorAll('#theme-toggle');
    themeToggleButtons.forEach(button => {
        const icon = button.querySelector('i');
        if (theme === 'dark') {
            icon.classList.remove('fa-sun');
            icon.classList.add('fa-moon');
        } else {
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
        }
    });
}