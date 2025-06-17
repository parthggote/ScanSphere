document.addEventListener('DOMContentLoaded', function() {
    // Set the current year in the footer
    const currentYearElements = document.querySelectorAll('#current-year');
    currentYearElements.forEach(element => {
        element.textContent = new Date().getFullYear();
    });

    // Check if user is logged in and update header
    updateUserHeaderDisplay();

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
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const submitButton = this.querySelector('button[type="submit"]');
            submitButton.classList.add('loading');
            
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            
            // Make actual API call to login endpoint
            fetch('http://localhost:5001/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email, password })
            })
            .then(response => response.json())
            .then(data => {
                submitButton.classList.remove('loading');
                
                if (data.success) {
                    // Store token in localStorage for future API calls
                    localStorage.setItem('token', data.token);
                    localStorage.setItem('user', JSON.stringify(data.user));
                    
                    // Direct redirect to the Flask app - simplified to avoid URL parsing issues
                    console.log('Redirecting to Flask app...');
                    window.location.href = data.redirectUrl;
                } else {
                    // Show error message
                    const errorMessage = data.message || 'Login failed. Please check your credentials.';
                    alert(errorMessage);
                }
            })
            .catch(error => {
                submitButton.classList.remove('loading');
                console.error('Login error:', error);
                alert('Login failed. Please try again later.');
            });
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
        
        signupForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            if (!validatePasswords()) {
                return;
            }
            
            const submitButton = this.querySelector('button[type="submit"]');
            submitButton.classList.add('loading');
            
            // Simulate signup process
            setTimeout(() => {
                submitButton.classList.remove('loading');
                window.location.href = 'login.html';
            }, 1500);
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

// Helper function to update header based on user login status
function updateUserHeaderDisplay() {
    // Check for auth buttons container (main header)
    const authButtonsContainer = document.querySelector('.auth-buttons');
    // Check for user profile container (direct container)
    const userProfileContainer = document.getElementById('user-profile-container');
    
    // Exit if neither container is found
    if (!authButtonsContainer && !userProfileContainer) return;
    
    // Check if user is logged in by looking for user data in localStorage
    const token = localStorage.getItem('token');
    
    if (token) {
        // Fetch current user data from API
        fetch('http://localhost:5001/api/user', {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success && data.user) {
                const user = data.user;
                const username = user.username || user.email.split('@')[0]; // Fallback to email username if no username
                
                // If we have a direct user profile container, update it
                if (userProfileContainer) {
                    if (userProfileContainer.classList.contains('user-profile-sidebar')) {
                        // Sidebar display
                        userProfileContainer.innerHTML = `
                            <div class="user-profile-info">
                                <div class="user-avatar">
                                    <i class="fas fa-user-circle"></i>
                                </div>
                                <div class="user-details">
                                    <div class="username">${username}</div>
                                    <div class="user-role">User</div>
                                </div>
                            </div>
                            <button id="logout-btn" class="logout-button">
                                <i class="fas fa-sign-out-alt"></i>
                                <span>Logout</span>
                            </button>
                        `;
                    } else {
                        // Header display
                        userProfileContainer.innerHTML = `
                            <div class="user-profile-info">
                                <div class="user-avatar">
                                    <i class="fas fa-user-circle"></i>
                                </div>
                                <div class="username">${username}</div>
                            </div>
                            <button id="logout-btn" class="logout-button">
                                <i class="fas fa-sign-out-alt"></i>
                            </button>
                        `;
                    }
                }
                
                // If we have auth buttons container, replace with user profile
                if (authButtonsContainer) {
                    authButtonsContainer.innerHTML = `
                        <div class="user-profile">
                            <div class="user-icon">
                                <i class="fas fa-user-circle"></i>
                            </div>
                            <span class="username">${username}</span>
                            <button id="logout-btn" class="btn-text">
                                <i class="fas fa-sign-out-alt"></i>
                            </button>
                        </div>
                    `;
                }
                
                // Add logout functionality to all logout buttons
                const logoutBtns = document.querySelectorAll('#logout-btn');
                logoutBtns.forEach(btn => {
                    btn.addEventListener('click', function() {
                        // Clear user data from storage
                        localStorage.removeItem('user');
                        localStorage.removeItem('token');
                        
                        // Redirect to home page
                        window.location.href = 'index.html';
                    });
                });
            } else {
                // If API call fails, try to use cached user data
                const userData = localStorage.getItem('user');
                if (userData) {
                    try {
                        const user = JSON.parse(userData);
                        const username = user.username || user.email.split('@')[0];
                        
                        // Update user interface with cached data
                        updateUIWithUserData(username, userProfileContainer, authButtonsContainer);
                    } catch (error) {
                        console.error('Error parsing cached user data:', error);
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error fetching user data:', error);
            
            // If API call fails, try to use cached user data
            const userData = localStorage.getItem('user');
            if (userData) {
                try {
                    const user = JSON.parse(userData);
                    const username = user.username || user.email.split('@')[0];
                    
                    // Update user interface with cached data
                    updateUIWithUserData(username, userProfileContainer, authButtonsContainer);
                } catch (error) {
                    console.error('Error parsing cached user data:', error);
                }
            }
        });
    }
}

// Helper function to update UI elements with user data
function updateUIWithUserData(username, userProfileContainer, authButtonsContainer) {
    // Generate initials from the username
    let initials = '';
    const nameParts = username.split(/[\s_\-\.]+/); // Split by space, underscore, dash or dot
    if (nameParts.length >= 2) {
        // Take first letter of first and last part
        initials = (nameParts[0][0] + nameParts[nameParts.length-1][0]).toUpperCase();
    } else if (username.length >= 2) {
        // Take first two letters if single name
        initials = username.substring(0, 2).toUpperCase();
    } else if (username.length === 1) {
        // If only one character
        initials = username.toUpperCase();
    } else {
        // Fallback if no name available
        initials = 'U';
    }
    
    // Generate a consistent color based on the username
    const colors = [
        '#FF5630', '#FF8B00', '#FFC400', '#36B37E', 
        '#00B8D9', '#6554C0', '#8777D9', '#998DD9', 
        '#C0B6F2', '#4C9AFF', '#0052CC', '#0747A6'
    ];
    const colorIndex = username.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % colors.length;
    const bgColor = colors[colorIndex];
    
    // If we have a direct user profile container, update it
    if (userProfileContainer) {
        if (userProfileContainer.classList.contains('user-profile-sidebar')) {
            // Sidebar display
            userProfileContainer.innerHTML = `
                <div class="user-profile-info">
                    <div class="user-avatar" style="background-color: ${bgColor}; color: white; display: flex; justify-content: center; align-items: center; width: 40px; height: 40px; border-radius: 50%; font-weight: bold;">
                        ${initials}
                    </div>
                    <div class="user-details">
                        <div class="username">${username}</div>
                        <div class="user-role">User</div>
                    </div>
                </div>
                <button id="logout-btn" class="logout-button">
                    <i class="fas fa-sign-out-alt"></i>
                    <span>Logout</span>
                </button>
            `;
        } else {
            // Header display
            userProfileContainer.innerHTML = `
                <div class="user-profile-info">
                    <div class="user-avatar" style="background-color: ${bgColor}; color: white; display: flex; justify-content: center; align-items: center; width: 32px; height: 32px; border-radius: 50%; font-weight: bold;">
                        ${initials}
                    </div>
                    <div class="username">${username}</div>
                </div>
                <button id="logout-btn" class="logout-button">
                    <i class="fas fa-sign-out-alt"></i>
                </button>
            `;
        }
    }
    
    // If we have auth buttons container, replace with user profile
    if (authButtonsContainer) {
        authButtonsContainer.innerHTML = `
            <div class="user-profile">
                <div class="user-icon" style="background-color: ${bgColor}; color: white; display: flex; justify-content: center; align-items: center; width: 32px; height: 32px; border-radius: 50%; font-weight: bold;">
                    ${initials}
                </div>
                <span class="username">${username}</span>
                <button id="logout-btn" class="btn-text">
                    <i class="fas fa-sign-out-alt"></i>
                </button>
            </div>
        `;
    }
    
    // Add logout functionality to all logout buttons
    const logoutBtns = document.querySelectorAll('#logout-btn');
    logoutBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            // Clear user data from storage
            localStorage.removeItem('user');
            localStorage.removeItem('token');
            
            // Redirect to home page
            window.location.href = 'index.html';
        });
    });
}