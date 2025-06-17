// History Page JavaScript

document.addEventListener('DOMContentLoaded', () => {
    // Initialize UI
    updateUserProfileDisplay();
    initializeHistorySidebar();
    setupTimeTravel();
    setupVoiceNotes();
    
    // Check if scan history exists, if not, add sample data for testing
    const history = JSON.parse(localStorage.getItem('scanHistory') || '[]');
    if (history.length === 0) {
        const sampleHistory = [
            {
                filename: 'sample1',
                timestamp: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(), // 30 days ago
                image: 'https://dummyimage.com/600x600/000/fff&text=Scan+1', // Better placeholder
                prediction: {
                    class: 'Meningioma',
                    confidence: 85.2
                },
                analysis: {
                    tumor_size: 15.8,
                    area: 196.3,
                    location: 'Frontal lobe'
                }
            },
            {
                filename: 'sample2',
                timestamp: new Date().toISOString(), // now
                image: 'https://dummyimage.com/600x600/000/fff&text=Scan+2', // Better placeholder
                prediction: {
                    class: 'Meningioma',
                    confidence: 92.7
                },
                analysis: {
                    tumor_size: 17.4,
                    area: 237.8,
                    location: 'Frontal lobe'
                }
            }
        ];
        localStorage.setItem('scanHistory', JSON.stringify(sampleHistory));
        console.log("Added sample scan history data:", sampleHistory);
    }
    
    // Ensure the metrics containers exist
    ensureMetricsContainers();
    
    loadScanOptions();
});

// Function to update user profile display (same as in index.html)
function updateUserProfileDisplay() {
    const userProfileContainer = document.getElementById('user-profile-container');
    if (!userProfileContainer) return;
    
    // Check for auth token in localStorage
    const token = localStorage.getItem('token');
    
    if (token) {
        // Try to get user data from server
        fetch('/api/user', {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success && data.user) {
                displayUserProfile(data.user, userProfileContainer);
            } else {
                // Fall back to stored data
                const userData = localStorage.getItem('user');
                if (userData) {
                    try {
                        const user = JSON.parse(userData);
                        displayUserProfile(user, userProfileContainer);
                    } catch (error) {
                        console.error('Error parsing user data:', error);
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error fetching user data:', error);
            // Fall back to stored data
            const userData = localStorage.getItem('user');
            if (userData) {
                try {
                    const user = JSON.parse(userData);
                    displayUserProfile(user, userProfileContainer);
                } catch (error) {
                    console.error('Error parsing user data:', error);
                }
            }
        });
    } else {
        // Show default guest user
        userProfileContainer.innerHTML = `
            <div class="user-profile-info">
                <div class="user-avatar default-avatar">
                    <i class="fas fa-user"></i>
                </div>
                <div class="username">Guest</div>
            </div>
            <button id="login-btn" class="login-button">
                <i class="fas fa-sign-in-alt"></i>
            </button>
        `;
        
        // Add login button functionality
        const loginBtn = document.getElementById('login-btn');
        if (loginBtn) {
            loginBtn.addEventListener('click', () => {
                window.location.href = '/login.html';
            });
        }
    }
}

// Helper function to display user profile (same as in index.html)
function displayUserProfile(user, container) {
    const username = user.username || user.email.split('@')[0];
    const profileImage = user.profileImage || null;
    
    // Generate initials from the username (for fallback)
    let initials = '';
    if (username) {
        initials = username.charAt(0).toUpperCase();
    } else {
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
    
    // Create avatar content
    let avatarContent;
    if (profileImage) {
        avatarContent = `<img src="${profileImage}" alt="${initials}" class="profile-image">`;
    } else {
        avatarContent = initials;
    }
    
    // Header display
    container.innerHTML = `
        <div class="user-profile-info">
            <div class="user-avatar" style="background-color: ${profileImage ? 'transparent' : bgColor}; color: white; display: flex; justify-content: center; align-items: center; width: 32px; height: 32px; border-radius: 50%; font-weight: bold;">
                ${avatarContent}
            </div>
            <div class="username">${username}</div>
        </div>
        <button id="logout-btn" class="logout-button">
            <i class="fas fa-sign-out-alt"></i>
        </button>
    `;
    
    // Add logout functionality
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function() {
            // Clear user data
            localStorage.removeItem('user');
            localStorage.removeItem('token');
            
            // Redirect to index (2).html page
            window.location.href = '/index (2).html';
        });
    }
}

// Function to initialize history sidebar
function initializeHistorySidebar() {
    // Get scan history from localStorage (if available)
    const history = JSON.parse(localStorage.getItem('scanHistory') || '[]');
    
    const historyContainer = document.querySelector('.scan-history');
    if (!historyContainer) return;
    
    // Clear existing items except title
    const items = historyContainer.querySelectorAll('.scan-item');
    items.forEach(item => {
        if (!item.classList.contains('no-scans')) {
            item.remove();
        }
    });
    
    // Add each scan to the sidebar
    if (history.length > 0) {
        // Remove the "No recent scans" message if it exists
        const noScansItem = historyContainer.querySelector('.no-scans');
        if (noScansItem) {
            noScansItem.remove();
        }
        
        history.forEach(scan => {
            const item = document.createElement('div');
            item.className = 'scan-item';
            item.innerHTML = `
                <i class="fas fa-file-medical"></i>
                <span>Scan_${scan.filename}</span>
            `;
            
            // Make the item clickable to view the scan
            item.addEventListener('click', () => {
                window.location.href = `/view-scan?filename=${encodeURIComponent(scan.filename)}`;
            });
            
            historyContainer.appendChild(item);
        });
    } else {
        // Add "No recent scans" message if there are no scans
        const noScansItem = document.createElement('div');
        noScansItem.className = 'scan-item no-scans';
        noScansItem.innerHTML = `
            <i class="fas fa-file-medical"></i>
            <span>No recent scans</span>
        `;
        historyContainer.appendChild(noScansItem);
    }
}

// Function to load scan options
function loadScanOptions() {
    // Get scan history from localStorage
    const history = JSON.parse(localStorage.getItem('scanHistory') || '[]');
    
    const scan1Select = document.getElementById('scan1');
    const scan2Select = document.getElementById('scan2');
    const compareBtn = document.getElementById('compare-btn');
    
    if (!scan1Select || !scan2Select) return;
    
    // Clear existing options
    scan1Select.innerHTML = '<option value="">Select a scan...</option>';
    scan2Select.innerHTML = '<option value="">Select a scan...</option>';
    
    // Sort scans by date (oldest first)
    history.sort((a, b) => {
        return new Date(a.timestamp) - new Date(b.timestamp);
    });
    
    // Add scan options
    history.forEach((scan, index) => {
        const date = new Date(scan.timestamp).toLocaleDateString();
        
        // Create option elements
        const option1 = document.createElement('option');
        option1.value = index;
        option1.textContent = `Scan ${index + 1} (${date})`;
        scan1Select.appendChild(option1);
        
        const option2 = document.createElement('option');
        option2.value = index;
        option2.textContent = `Scan ${index + 1} (${date})`;
        scan2Select.appendChild(option2);
    });
    
    // Enable compare button when both scans are selected
    function checkSelections() {
        if (scan1Select.value && scan2Select.value && scan1Select.value !== scan2Select.value) {
            compareBtn.disabled = false;
        } else {
            compareBtn.disabled = true;
        }
    }
    
    scan1Select.addEventListener('change', checkSelections);
    scan2Select.addEventListener('change', checkSelections);
    
    // Handle compare button click
    compareBtn.addEventListener('click', () => {
        const scan1Index = parseInt(scan1Select.value);
        const scan2Index = parseInt(scan2Select.value);
        
        console.log("Compare button clicked - Selected indices:", scan1Index, scan2Index);
        
        if (isNaN(scan1Index) || isNaN(scan2Index)) {
            console.error("Invalid scan indices", { scan1Index, scan2Index });
            return;
        }
        
        // Get the scan data
        const scan1 = history[scan1Index];
        const scan2 = history[scan2Index];
        
        console.log("Retrieved scan data:", { scan1, scan2 });
        
        if (!scan1 || !scan2) {
            console.error("Failed to retrieve scan data", { scan1, scan2 });
            return;
        }
        
        // Clear any existing localStorage for this session to prevent issues
        sessionStorage.removeItem('currentComparison');
        sessionStorage.setItem('currentComparison', JSON.stringify({ scan1Index, scan2Index }));
        
        // Show comparison section - WITH ERROR HANDLING
        const comparisonContainer = document.getElementById('comparison-container');
        if (!comparisonContainer) {
            console.error("Comparison container not found in DOM");
            return;
        }
        
        // Safely remove d-none class and set display style
        comparisonContainer.classList.remove('d-none');
        if (comparisonContainer.style !== undefined) {
            comparisonContainer.style.display = 'block';
        }
        console.log("Showing comparison container:", comparisonContainer);
        
        // Ensure metrics containers exist
        if (!ensureMetricsContainers()) {
            console.error("Failed to ensure metrics containers exist - may cause display issues");
        }
        
        // Scroll to the comparison container only if it exists and scrollIntoView is available
        if (comparisonContainer && typeof comparisonContainer.scrollIntoView === 'function') {
            comparisonContainer.scrollIntoView({ behavior: 'smooth' });
        }
        
        // Initialize comparison UI with a small delay to ensure UI is ready
        setTimeout(() => {
            initializeComparison(scan1, scan2);
        }, 100);
    });
}

// Function to setup the time-travel comparison UI
function setupTimeTravel() {
    const slider = document.getElementById('comparison-slider');
    const container = document.getElementById('comparison-container');
    const overlay = document.querySelector('.img-overlay');
    const imgContainer = document.querySelector('.img-container');
    
    // Log the presence of all required elements
    console.log("Setting up time travel UI with elements:", {
        slider: !!slider,
        container: !!container,
        overlay: !!overlay,
        imgContainer: !!imgContainer
    });
    
    if (!slider || !container || !overlay || !imgContainer) {
        console.error("Missing comparison UI elements:", {
            slider: !!slider,
            container: !!container,
            overlay: !!overlay,
            imgContainer: !!imgContainer
        });
        return;
    }
    
    // Add mouse drag functionality
    let isDragging = false;
    
    // Safely add event listeners with try-catch blocks
    try {
        slider.addEventListener('mousedown', () => {
            isDragging = true;
        });
        
        document.addEventListener('mouseup', () => {
            isDragging = false;
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            
            try {
                // Get container bounds
                const rect = imgContainer.getBoundingClientRect();
                
                // Calculate position
                let position = (e.clientX - rect.left) / rect.width;
                position = Math.max(0, Math.min(1, position));
                
                // Update overlay width - with null checks
                if (overlay && typeof overlay.style !== 'undefined') {
                    overlay.style.width = `${position * 100}%`;
                }
                
                // Update slider position - with null checks
                if (slider && typeof slider.style !== 'undefined') {
                    slider.style.left = `${position * 100}%`;
                }
            } catch (error) {
                console.error("Error during mouse move:", error);
                isDragging = false;
            }
        });
        
        // Add touch functionality for mobile
        slider.addEventListener('touchstart', () => {
            isDragging = true;
        });
        
        document.addEventListener('touchend', () => {
            isDragging = false;
        });
        
        document.addEventListener('touchmove', (e) => {
            if (!isDragging) return;
            
            try {
                // Prevent scrolling
                e.preventDefault();
                
                // Get container bounds
                const rect = imgContainer.getBoundingClientRect();
                
                // Calculate position
                let position = (e.touches[0].clientX - rect.left) / rect.width;
                position = Math.max(0, Math.min(1, position));
                
                // Update overlay width - with null checks
                if (overlay && typeof overlay.style !== 'undefined') {
                    overlay.style.width = `${position * 100}%`;
                }
                
                // Update slider position - with null checks
                if (slider && typeof slider.style !== 'undefined') {
                    slider.style.left = `${position * 100}%`;
                }
            } catch (error) {
                console.error("Error during touch move:", error);
                isDragging = false;
            }
        });
        
        console.log("Time travel UI setup completed successfully");
    } catch (error) {
        console.error("Error setting up time travel UI:", error);
    }
}

// Function to initialize the comparison UI with scan data
function initializeComparison(scan1, scan2) {
    // Ensure metrics containers exist before proceeding
    if (!ensureMetricsContainers()) {
        console.error("Failed to ensure metrics containers exist - attempting to continue");
    }
    
    // Get all required elements
    const previousScan = document.getElementById('previous-scan');
    const currentScan = document.getElementById('current-scan');
    const tumorMetrics = document.getElementById('tumor-metrics');
    const analysisMetrics = document.getElementById('analysis-metrics');
    const imgContainer = document.querySelector('.img-container');
    const comparisonContainer = document.getElementById('comparison-container');
    const imgOverlay = document.querySelector('.img-overlay');
    const comparisonSlider = document.getElementById('comparison-slider');
    
    console.log("Initializing comparison UI with elements:", {
        previousScan: !!previousScan,
        currentScan: !!currentScan,
        tumorMetrics: !!tumorMetrics,
        analysisMetrics: !!analysisMetrics,
        imgContainer: !!imgContainer,
        comparisonContainer: !!comparisonContainer,
        imgOverlay: !!imgOverlay,
        comparisonSlider: !!comparisonSlider
    });
    
    if (!previousScan || !currentScan) {
        console.error("Missing image elements - cannot continue comparison");
        return;
    }
    
    if (!tumorMetrics || !analysisMetrics) {
        console.error("Missing metrics elements - will attempt to recreate");
        if (!ensureMetricsContainers()) {
            console.error("Failed to recreate metrics containers - cannot continue");
            return;
        }
        
        // Try again after recreation
        const tumorMetrics = document.getElementById('tumor-metrics');
        const analysisMetrics = document.getElementById('analysis-metrics');
        
        if (!tumorMetrics || !analysisMetrics) {
            console.error("Still missing metrics elements after recreation - cannot continue");
            return;
        }
    }
    
    // Set image sources safely
    try {
        if (scan1 && scan1.image) {
            previousScan.src = scan1.image;
        } else if (scan1 && scan1.filename) {
            previousScan.src = `/uploads/${scan1.filename}`;
        } else {
            console.error("No valid image source for previous scan");
            previousScan.src = 'https://dummyimage.com/400x400/000/fff&text=No+Image';
        }
        
        if (scan2 && scan2.image) {
            currentScan.src = scan2.image;
        } else if (scan2 && scan2.filename) {
            currentScan.src = `/uploads/${scan2.filename}`;
        } else {
            console.error("No valid image source for current scan");
            currentScan.src = 'https://dummyimage.com/400x400/000/fff&text=No+Image';
        }
    } catch (error) {
        console.error("Error setting image sources:", error);
    }
    
    // Reset the slider and overlay position - with proper null checks
    if (imgOverlay && typeof imgOverlay.style !== 'undefined') {
        try {
            imgOverlay.style.width = '50%';
        } catch (error) {
            console.error("Error setting imgOverlay width:", error);
        }
    } else {
        console.warn("imgOverlay element is missing or has no style property");
    }
    
    if (comparisonSlider && typeof comparisonSlider.style !== 'undefined') {
        try {
            comparisonSlider.style.left = '50%';
        } catch (error) {
            console.error("Error setting comparisonSlider left position:", error);
        }
    } else {
        console.warn("comparisonSlider element is missing or has no style property");
    }
    
    // Calculate metrics for comparison
    try {
        const metrics = calculateMetrics(scan1, scan2);
        const overallStatus = determineOverallStatus(metrics);
        
        // Update the metrics UI if the elements exist
        if (tumorMetrics && analysisMetrics) {
            updateMetricsUI(metrics, tumorMetrics, analysisMetrics, overallStatus);
        }
    } catch (error) {
        console.error("Error calculating or displaying metrics:", error);
        
        // Display a fallback message if metrics calculation fails
        if (tumorMetrics) {
            tumorMetrics.innerHTML = '<div class="alert alert-warning">Unable to calculate metrics for these scans.</div>';
        }
        
        if (analysisMetrics) {
            analysisMetrics.innerHTML = '<div class="alert alert-warning">Analysis data unavailable.</div>';
        }
    }
}

// Function to calculate metrics for comparing scans
function calculateMetrics(scan1, scan2) {
    console.log("Calculating metrics for scans:", { 
        scan1: scan1 ? `id: ${scan1.filename}, timestamp: ${scan1.timestamp}` : 'undefined',
        scan2: scan2 ? `id: ${scan2.filename}, timestamp: ${scan2.timestamp}` : 'undefined'
    });
    
    // Safely extract values with fallbacks
    const safePrevSize = scan1 && scan1.analysis && scan1.analysis.tumor_size ? scan1.analysis.tumor_size : 'Unknown';
    const safeCurrSize = scan2 && scan2.analysis && scan2.analysis.tumor_size ? scan2.analysis.tumor_size : 'Unknown';
    const safePrevArea = scan1 && scan1.analysis && scan1.analysis.area ? scan1.analysis.area : 'Unknown';
    const safeCurrArea = scan2 && scan2.analysis && scan2.analysis.area ? scan2.analysis.area : 'Unknown';
    
    // Calculate size change if both values are available
    let sizeChange = 'Unknown';
    let areaChange = 'Unknown';
    let changeType = 'Unknown';
    
    if (safePrevSize !== 'Unknown' && safeCurrSize !== 'Unknown') {
        const sizeChangeValue = ((safeCurrSize - safePrevSize) / safePrevSize) * 100;
        sizeChange = `${Math.abs(sizeChangeValue).toFixed(1)}%`;
        
        if (sizeChangeValue > 2) {
            changeType = 'growth';
        } else if (sizeChangeValue < -2) {
            changeType = 'reduction';
        } else {
            changeType = 'stable';
        }
    }
    
    if (safePrevArea !== 'Unknown' && safeCurrArea !== 'Unknown') {
        const areaChangeValue = ((safeCurrArea - safePrevArea) / safePrevArea) * 100;
        areaChange = `${Math.abs(areaChangeValue).toFixed(1)}%`;
    }
    
    // Calculate time between scans
    let timeframe = 'Unknown';
    let daysBetween = 0;
    let growthRate = 'Unknown';
    
    if (scan1 && scan1.timestamp && scan2 && scan2.timestamp) {
        try {
            const date1 = new Date(scan1.timestamp);
            const date2 = new Date(scan2.timestamp);
            
            if (!isNaN(date1) && !isNaN(date2)) {
                timeframe = calculateTimeframe(date1, date2);
                daysBetween = calculateDaysBetween(date1, date2);
                
                // Calculate daily growth rate if size is known
                if (safePrevSize !== 'Unknown' && safeCurrSize !== 'Unknown' && daysBetween > 0) {
                    const dailyGrowth = (safeCurrSize - safePrevSize) / daysBetween;
                    growthRate = `${Math.abs(dailyGrowth).toFixed(2)} mm/day`;
                }
            }
        } catch (error) {
            console.error("Error calculating timeframe:", error);
        }
    }
    
    // Construct the metrics object
    return {
        previous: {
            size: {
                value: safePrevSize !== 'Unknown' ? `${safePrevSize} mm` : 'Unknown',
                raw: safePrevSize
            },
            area: {
                value: safePrevArea !== 'Unknown' ? `${safePrevArea} mm²` : 'Unknown',
                raw: safePrevArea
            },
            volume: {
                value: 'Unknown',
                raw: 'Unknown'
            },
            classification: scan1 && scan1.prediction && scan1.prediction.class ? scan1.prediction.class : 'Unknown',
            confidence: scan1 && scan1.prediction && scan1.prediction.confidence ? `${scan1.prediction.confidence.toFixed(1)}%` : 'Unknown',
            location: scan1 && scan1.analysis && scan1.analysis.location ? scan1.analysis.location : 'Unknown',
            density: 'Unknown',
            margin: 'Unknown',
            enhancement: 'Unknown',
            stage: 'Unknown'
        },
        current: {
            size: {
                value: safeCurrSize !== 'Unknown' ? `${safeCurrSize} mm` : 'Unknown',
                raw: safeCurrSize
            },
            area: {
                value: safeCurrArea !== 'Unknown' ? `${safeCurrArea} mm²` : 'Unknown',
                raw: safeCurrArea
            },
            volume: {
                value: 'Unknown',
                raw: 'Unknown'
            },
            classification: scan2 && scan2.prediction && scan2.prediction.class ? scan2.prediction.class : 'Unknown',
            confidence: scan2 && scan2.prediction && scan2.prediction.confidence ? `${scan2.prediction.confidence.toFixed(1)}%` : 'Unknown',
            location: scan2 && scan2.analysis && scan2.analysis.location ? scan2.analysis.location : 'Unknown',
            density: 'Unknown',
            margin: 'Unknown',
            enhancement: 'Unknown',
            stage: 'Unknown'
        },
        comparison: {
            size: sizeChange,
            area: areaChange,
            volume: 'Unknown',
            changeType: changeType,
            timeframe: timeframe,
            growthRate: growthRate
        }
    };
}

// Helper function to calculate timeframe in a human-readable format
function calculateTimeframe(date1, date2) {
    const diffTime = Math.abs(date2 - date1);
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays < 1) {
        return "Less than a day";
    } else if (diffDays === 1) {
        return "1 day";
    } else if (diffDays < 7) {
        return `${diffDays} days`;
    } else if (diffDays < 14) {
        return "1 week";
    } else if (diffDays < 30) {
        return `${Math.floor(diffDays / 7)} weeks`;
    } else if (diffDays < 60) {
        return "1 month";
    } else if (diffDays < 365) {
        return `${Math.floor(diffDays / 30)} months`;
    } else if (diffDays < 730) {
        return "1 year";
    } else {
        return `${Math.floor(diffDays / 365)} years`;
    }
}

// Helper function to calculate days between dates
function calculateDaysBetween(date1, date2) {
    const diffTime = Math.abs(date2 - date1);
    return Math.floor(diffTime / (1000 * 60 * 60 * 24));
}

// Function to determine the overall status based on metrics
function determineOverallStatus(metrics) {
    const changeType = metrics.comparison.changeType;
    const currConfidence = metrics.current.confidence;
    let confidenceValue = 0;
    
    if (currConfidence && currConfidence !== "Unknown") {
        confidenceValue = parseFloat(currConfidence);
    }
    
    // Check for significant growth with high confidence
    if (changeType === "growth" && confidenceValue > 85) {
        return {
            status: "concern",
            label: "Concerning Growth",
            description: "Significant tumor growth detected with high confidence. Medical attention recommended.",
            color: "danger"
        };
    }
    
    // Check for reduction with high confidence
    if (changeType === "reduction" && confidenceValue > 85) {
        return {
            status: "improving",
            label: "Improvement Detected",
            description: "Reduction in tumor size detected with high confidence. Treatment appears effective.",
            color: "success"
        };
    }
    
    // Check for stability with high confidence
    if (changeType === "stable" && confidenceValue > 85) {
        return {
            status: "stable",
            label: "Stable Condition",
            description: "No significant change in tumor metrics. Continue monitoring.",
            color: "warning"
        };
    }
    
    // Default status for uncertain cases
    return {
        status: "monitoring",
        label: "Continued Monitoring",
        description: "Changes detected but confidence levels suggest continued monitoring.",
        color: "info"
    };
}

// Function to update metrics UI sections
function updateMetricsUI(metrics, tumorMetricsEl, analysisMetricsEl, overallStatus) {
    console.log("Updating metrics UI with:", metrics);
    console.log("Elements exist:", { 
        tumorMetricsEl: !!tumorMetricsEl, 
        analysisMetricsEl: !!analysisMetricsEl,
        tumorMetricsElId: tumorMetricsEl ? tumorMetricsEl.id : 'missing',
        analysisMetricsElId: analysisMetricsEl ? analysisMetricsEl.id : 'missing'
    });
    
    if (!tumorMetricsEl || !analysisMetricsEl) {
        console.error("One or more metrics elements are missing in the DOM");
        // Try to recover them by ID
        tumorMetricsEl = document.getElementById('tumor-metrics');
        analysisMetricsEl = document.getElementById('analysis-metrics');
        
        if (!tumorMetricsEl || !analysisMetricsEl) {
            console.error("Failed to recover metrics elements");
            return;
        }
    }

    // Update tumor metrics
    const tumorHtml = `
        <div class="metric-item">
            <span class="metric-label">Previous Size:</span>
            <span class="metric-value">${metrics.previous.size.value}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Current Size:</span>
            <span class="metric-value">${metrics.current.size.value}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Size Change:</span>
            <span class="metric-value ${getChangeClass(metrics.comparison.changeType)}">
                ${metrics.comparison.size} ${metrics.comparison.changeType}
            </span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Previous Area:</span>
            <span class="metric-value">${metrics.previous.area.value}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Current Area:</span>
            <span class="metric-value">${metrics.current.area.value}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Area Change:</span>
            <span class="metric-value ${getChangeClass(metrics.comparison.changeType)}">
                ${metrics.comparison.area}
            </span>
        </div>
        ${metrics.previous.volume.value !== "Unknown" || metrics.current.volume.value !== "Unknown" ? `
        <div class="metric-item">
            <span class="metric-label">Previous Volume:</span>
            <span class="metric-value">${metrics.previous.volume.value}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Current Volume:</span>
            <span class="metric-value">${metrics.current.volume.value}</span>
        </div>
        ${metrics.comparison.volume !== "Unknown" ? `
        <div class="metric-item">
            <span class="metric-label">Volume Change:</span>
            <span class="metric-value ${getChangeClass(metrics.comparison.changeType)}">
                ${metrics.comparison.volume}
            </span>
        </div>
        ` : ''}
        ` : ''}
        <div class="metric-item">
            <span class="metric-label">Previous Confidence:</span>
            <span class="metric-value">${metrics.previous.confidence}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Current Confidence:</span>
            <span class="metric-value">${metrics.current.confidence}</span>
        </div>
        ${metrics.previous.density !== "Unknown" || metrics.current.density !== "Unknown" ? `
        <div class="metric-item">
            <span class="metric-label">Previous Density:</span>
            <span class="metric-value">${metrics.previous.density}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Current Density:</span>
            <span class="metric-value">${metrics.current.density}</span>
        </div>
        ` : ''}
    `;
    
    console.log("Generated tumor metrics HTML:", tumorHtml.substring(0, 100) + "...");
    
    // Force updating the metrics container
    try {
        tumorMetricsEl.innerHTML = tumorHtml;
        console.log("Updated tumor metrics HTML");
    } catch (e) {
        console.error("Error updating tumor metrics:", e);
    }
    
    // Update analysis metrics with comprehensive data
    const analysisHtml = `
        <div class="overall-status mb-3 p-2 rounded bg-${overallStatus.color}">
            <h5 class="text-white mb-1">${overallStatus.label}</h5>
            <p class="text-white mb-0 small">${overallStatus.description}</p>
        </div>
        
        <div class="metric-item">
            <span class="metric-label">Previous Classification:</span>
            <span class="metric-value">${metrics.previous.classification || "Unknown"}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Current Classification:</span>
            <span class="metric-value">${metrics.current.classification || "Unknown"}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Timeframe:</span>
            <span class="metric-value">${metrics.comparison.timeframe}</span>
        </div>
        ${metrics.comparison.growthRate ? `
        <div class="metric-item">
            <span class="metric-label">Growth Rate:</span>
            <span class="metric-value ${metrics.comparison.changeType === "growth" ? "text-danger" : "text-muted"}">
                ${metrics.comparison.growthRate}
            </span>
        </div>
        ` : ''}
        <div class="metric-item">
            <span class="metric-label">Location:</span>
            <span class="metric-value">${metrics.current.location}</span>
        </div>
        ${metrics.current.margin !== "Unknown" ? `
        <div class="metric-item">
            <span class="metric-label">Margin:</span>
            <span class="metric-value">${metrics.current.margin}</span>
        </div>
        ` : ''}
        ${metrics.current.enhancement !== "Unknown" ? `
        <div class="metric-item">
            <span class="metric-label">Enhancement:</span>
            <span class="metric-value">${metrics.current.enhancement}</span>
        </div>
        ` : ''}
        ${metrics.current.stage !== "Unknown" ? `
        <div class="metric-item">
            <span class="metric-label">Stage:</span>
            <span class="metric-value">${metrics.current.stage}</span>
        </div>
        ` : ''}
    `;
    
    console.log("Generated analysis metrics HTML:", analysisHtml.substring(0, 100) + "...");
    
    // Force updating the analysis metrics container
    try {
        analysisMetricsEl.innerHTML = analysisHtml;
        console.log("Updated analysis metrics HTML");
    } catch (e) {
        console.error("Error updating analysis metrics:", e);
    }
}

// Helper function to get CSS class for change type
function getChangeClass(changeType) {
    if (changeType === "growth") return "text-danger";
    if (changeType === "reduction") return "text-success";
    return "text-warning";
}

// Function to setup voice notes
function setupVoiceNotes() {
    const recordBtn = document.getElementById('voice-record-btn');
    const waveformEl = document.getElementById('waveform');
    const transcriptionEl = document.getElementById('transcription');
    const editBtn = document.getElementById('edit-transcript-btn');
    const saveBtn = document.getElementById('save-notes-btn');
    
    if (!recordBtn || !waveformEl || !transcriptionEl || !editBtn || !saveBtn) return;
    
    // Check if browser supports speech recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
        transcriptionEl.innerHTML = '<div class="alert alert-warning">Speech recognition is not supported in your browser. Please use Chrome, Edge, or Safari.</div>';
        return;
    }
    
    // Create speech recognition object
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    
    // Initialize WaveSurfer.js for waveform visualization
    const wavesurfer = WaveSurfer.create({
        container: waveformEl,
        waveColor: 'rgba(255, 255, 255, 0.3)',
        progressColor: 'var(--accent-color)',
        cursorColor: 'transparent',
        barWidth: 2,
        barGap: 1,
        height: 60,
        responsive: true
    });
    
    // Variables for recording state
    let isRecording = false;
    let mediaRecorder;
    let audioChunks = [];
    let finalTranscript = '';
    
    // Toggle recording
    recordBtn.addEventListener('click', () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });
    
    // Start recording function
    function startRecording() {
        // Reset previous data
        audioChunks = [];
        finalTranscript = '';
        transcriptionEl.innerHTML = '<div class="text-muted">Listening...</div>';
        
        // Request microphone access
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                // Mark as recording
                isRecording = true;
                recordBtn.classList.add('recording');
                
                // Create media recorder
                mediaRecorder = new MediaRecorder(stream);
                
                // Start speech recognition
                recognition.start();
                
                // Setup media recorder
                mediaRecorder.ondataavailable = (e) => {
                    audioChunks.push(e.data);
                };
                
                mediaRecorder.onstop = () => {
                    // Create audio blob
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    
                    // Create audio URL
                    const audioUrl = URL.createObjectURL(audioBlob);
                    
                    // Load audio into wavesurfer
                    wavesurfer.load(audioUrl);
                    
                    // Show edit and save buttons
                    editBtn.style.display = 'inline-block';
                    saveBtn.style.display = 'inline-block';
                };
                
                // Start recording
                mediaRecorder.start();
                
                // Simulate waveform animation (would be better with real audio analysis)
                animateWaveform();
            })
            .catch(error => {
                console.error('Error accessing microphone:', error);
                transcriptionEl.innerHTML = `<div class="alert alert-danger">Error accessing microphone: ${error.message}</div>`;
            });
    }
    
    // Stop recording function
    function stopRecording() {
        isRecording = false;
        recordBtn.classList.remove('recording');
        
        // Stop speech recognition
        recognition.stop();
        
        // Stop media recorder
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        
        // Stop all tracks in the stream
        if (mediaRecorder && mediaRecorder.stream) {
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }
    
    // Handle speech recognition results
    recognition.onresult = (event) => {
        let interimTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            
            if (event.results[i].isFinal) {
                finalTranscript += transcript + ' ';
            } else {
                interimTranscript += transcript;
            }
        }
        
        // Update transcription element
        transcriptionEl.innerHTML = `
            <div>${finalTranscript}</div>
            <div><em class="text-muted">${interimTranscript}</em></div>
        `;
    };
    
    // Handle speech recognition errors
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        
        if (event.error === 'no-speech') {
            transcriptionEl.innerHTML = '<div class="alert alert-info">No speech detected. Please try again.</div>';
        } else {
            transcriptionEl.innerHTML = `<div class="alert alert-danger">Error: ${event.error}</div>`;
        }
        
        stopRecording();
    };
    
    // Handle edit transcript button
    editBtn.addEventListener('click', () => {
        transcriptionEl.contentEditable = true;
        transcriptionEl.focus();
        
        // Add a save option
        transcriptionEl.innerHTML = `<div>${finalTranscript}</div>
            <div class="text-muted mt-2">Click anywhere outside this area when finished editing</div>`;
        
        // Save on blur
        transcriptionEl.addEventListener('blur', function saveEdit() {
            transcriptionEl.contentEditable = false;
            finalTranscript = transcriptionEl.innerText.trim();
            transcriptionEl.innerHTML = `<div>${finalTranscript}</div>`;
            transcriptionEl.removeEventListener('blur', saveEdit);
        }, { once: true });
    });
    
    // Handle save notes button
    saveBtn.addEventListener('click', () => {
        if (!finalTranscript.trim()) {
            alert('Please record or type some notes first.');
            return;
        }
        
        // Get the currently selected scans
        const scan1Index = document.getElementById('scan1').value;
        const scan2Index = document.getElementById('scan2').value;
        
        if (!scan1Index || !scan2Index) {
            alert('Please select two scans to compare before saving notes.');
            return;
        }
        
        // Create the audio element for the diagnostic report
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // Save the audio and transcript to the report section
        addToReport(finalTranscript, audioUrl);
        
        // Also save to localStorage for future reference
        const notes = JSON.parse(localStorage.getItem('voiceNotes') || '[]');
        
        notes.push({
            text: finalTranscript,
            date: new Date().toISOString(),
            audioUrl: audioUrl,
            scans: [scan1Index, scan2Index].filter(Boolean)
        });
        
        localStorage.setItem('voiceNotes', JSON.stringify(notes));
        
        // Show success message
        alert('Notes added to diagnostic report!');
        
        // Reset form
        finalTranscript = '';
        transcriptionEl.innerHTML = '<div class="text-muted">Click the microphone button to start recording</div>';
        editBtn.style.display = 'none';
        saveBtn.style.display = 'none';
        wavesurfer.empty();
    });
    
    // Function to add notes to the diagnostic report
    function addToReport(transcript, audioUrl) {
        // Check if diagnostic report section exists, if not create it
        let reportSection = document.getElementById('diagnostic-report-section');
        if (!reportSection) {
            reportSection = document.createElement('div');
            reportSection.id = 'diagnostic-report-section';
            reportSection.className = 'results-card mt-4';
            
            // Create the section header
            reportSection.innerHTML = `
                <h4><i class="fas fa-file-medical-alt"></i> Diagnostic Report</h4>
                <p class="text-muted">Scan comparison analysis and doctor's notes</p>
                <div id="diagnostic-report-content" class="report-content"></div>
                <div class="text-center mt-3">
                    <button id="export-report-btn" class="action-button">
                        <i class="fas fa-download"></i>
                        <span>Export Report</span>
                    </button>
                    <button id="print-report-btn" class="action-button">
                        <i class="fas fa-print"></i>
                        <span>Print Report</span>
                    </button>
                </div>
            `;
            
            // Add it before the voice notes section
            const voiceNotesSection = document.getElementById('voice-notes-section');
            voiceNotesSection.parentNode.insertBefore(reportSection, voiceNotesSection);
            
            // Add event listeners for export and print buttons
            document.getElementById('export-report-btn').addEventListener('click', exportReport);
            document.getElementById('print-report-btn').addEventListener('click', printReport);
        }
        
        // Get the report content container
        const reportContent = document.getElementById('diagnostic-report-content');
        
        // Get scan information
        const scan1Index = document.getElementById('scan1').value;
        const scan2Index = document.getElementById('scan2').value;
        const history = JSON.parse(localStorage.getItem('scanHistory') || '[]');
        const scan1 = history[scan1Index];
        const scan2 = history[scan2Index];
        
        // Create a timestamp
        const timestamp = new Date().toLocaleString();
        
        // Create the note element
        const noteElement = document.createElement('div');
        noteElement.className = 'doctor-note mb-4';
        noteElement.innerHTML = `
            <div class="note-header d-flex justify-content-between align-items-center">
                <h5>Doctor's Note - ${timestamp}</h5>
                <span class="badge bg-primary">Voice Note</span>
            </div>
            <div class="note-content">
                <div class="audio-player mb-3">
                    <audio controls>
                        <source src="${audioUrl}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                </div>
                <div class="transcript">
                    <p><strong>Transcript:</strong></p>
                    <p>${transcript}</p>
                </div>
                <div class="scan-info mt-3">
                    <p><small class="text-muted">Referring to scans from: 
                    ${scan1 ? new Date(scan1.timestamp).toLocaleDateString() : 'Unknown'} and 
                    ${scan2 ? new Date(scan2.timestamp).toLocaleDateString() : 'Unknown'}</small></p>
                </div>
            </div>
        `;
        
        // Add to the report
        reportContent.appendChild(noteElement);
        
        // Scroll to the report section
        reportSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Function to export the report as PDF or HTML
    function exportReport() {
        const reportContent = document.getElementById('diagnostic-report-content');
        if (!reportContent || !reportContent.innerHTML.trim()) {
            alert('No report content to export.');
            return;
        }
        
        // Create a blob with the HTML content
        const htmlContent = `
            <!DOCTYPE html>
            <html>
            <head>
                <title>Diagnostic Report - ${new Date().toLocaleDateString()}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                    .report-header { text-align: center; margin-bottom: 30px; }
                    .doctor-note { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                    .note-header { margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
                    .transcript { margin-top: 15px; }
                    .scan-info { margin-top: 20px; font-size: 0.9em; color: #777; }
                </style>
            </head>
            <body>
                <div class="report-header">
                    <h1>Brain Tumor Analysis</h1>
                    <h2>Diagnostic Report</h2>
                    <p>Generated on ${new Date().toLocaleString()}</p>
                </div>
                ${reportContent.innerHTML}
            </body>
            </html>
        `;
        
        const blob = new Blob([htmlContent], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        
        // Create a link to download
        const a = document.createElement('a');
        a.href = url;
        a.download = `diagnostic-report-${new Date().toISOString().split('T')[0]}.html`;
        a.click();
        
        // Clean up
        URL.revokeObjectURL(url);
    }
    
    // Function to print the report
    function printReport() {
        const reportContent = document.getElementById('diagnostic-report-content');
        if (!reportContent || !reportContent.innerHTML.trim()) {
            alert('No report content to print.');
            return;
        }
        
        // Create a new window with just the report content
        const printWindow = window.open('', '_blank');
        printWindow.document.write(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>Print Diagnostic Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                    .report-header { text-align: center; margin-bottom: 30px; }
                    .doctor-note { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                    .note-header { margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
                    .transcript { margin-top: 15px; }
                    .scan-info { margin-top: 20px; font-size: 0.9em; color: #777; }
                    @media print {
                        button { display: none; }
                    }
                </style>
            </head>
            <body>
                <div class="report-header">
                    <h1>Brain Tumor Analysis</h1>
                    <h2>Diagnostic Report</h2>
                    <p>Generated on ${new Date().toLocaleString()}</p>
                </div>
                ${reportContent.innerHTML}
                <script>
                    // Print and close automatically
                    window.onload = function() {
                        window.print();
                        setTimeout(function() { window.close(); }, 500);
                    };
                </script>
            </body>
            </html>
        `);
        printWindow.document.close();
    }
    
    // Simulate waveform animation
    function animateWaveform() {
        if (!isRecording) return;
        
        // Generate random waveform data (in a real app, would use actual audio analysis)
        const data = Array.from({ length: 50 }, () => Math.random() * 0.5 + 0.1);
        
        // Load data into wavesurfer
        wavesurfer.load(
            'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQQAAAA='
        );
        
        // Repeat while recording
        setTimeout(() => {
            if (isRecording) {
                animateWaveform();
            }
        }, 500);
    }
}

// Function to ensure metrics containers exist
function ensureMetricsContainers() {
    const comparisonContainer = document.getElementById('comparison-container');
    if (!comparisonContainer) {
        console.error("Comparison container is missing");
        return false;
    }
    
    // Check if metrics container exists
    let metricsContainer = comparisonContainer.querySelector('.metrics-container');
    if (!metricsContainer) {
        console.log("Metrics container is missing - creating it");
        
        try {
            metricsContainer = document.createElement('div');
            metricsContainer.className = 'metrics-container';
            
            // Find where to add it
            const row = comparisonContainer.querySelector('.row:nth-child(2)');
            if (!row) {
                // Create a new row if it doesn't exist
                try {
                    const newRow = document.createElement('div');
                    newRow.className = 'row';
                    newRow.innerHTML = '<div class="col-lg-8 mx-auto"></div>';
                    comparisonContainer.appendChild(newRow);
                    const col = newRow.querySelector('.col-lg-8');
                    if (col) {
                        col.appendChild(metricsContainer);
                    } else {
                        console.error("Failed to find column in new row");
                        return false;
                    }
                } catch (error) {
                    console.error("Error creating new row:", error);
                    return false;
                }
            } else {
                // Use existing row
                const col = row.querySelector('.col-lg-8');
                if (col) {
                    col.appendChild(metricsContainer);
                } else {
                    try {
                        row.innerHTML = '<div class="col-lg-8 mx-auto"></div>';
                        const newCol = row.querySelector('.col-lg-8');
                        if (newCol) {
                            newCol.appendChild(metricsContainer);
                        } else {
                            console.error("Failed to create column in existing row");
                            return false;
                        }
                    } catch (error) {
                        console.error("Error setting row innerHTML:", error);
                        return false;
                    }
                }
            }
        } catch (error) {
            console.error("Error creating metrics container:", error);
            return false;
        }
    }
    
    // Check if tumor metrics section exists
    let tumorSection = metricsContainer.querySelector('.metrics-section:nth-child(1)');
    if (!tumorSection) {
        console.log("Tumor metrics section is missing - creating it");
        try {
            tumorSection = document.createElement('div');
            tumorSection.className = 'metrics-section';
            tumorSection.innerHTML = `
                <h4 class="metrics-title">Tumor Metrics</h4>
                <div id="tumor-metrics" class="metrics-data"></div>
            `;
            metricsContainer.appendChild(tumorSection);
        } catch (error) {
            console.error("Error creating tumor metrics section:", error);
            return false;
        }
    }
    
    // Check if analysis metrics section exists
    let analysisSection = metricsContainer.querySelector('.metrics-section:nth-child(2)');
    if (!analysisSection) {
        console.log("Analysis metrics section is missing - creating it");
        try {
            analysisSection = document.createElement('div');
            analysisSection.className = 'metrics-section';
            analysisSection.innerHTML = `
                <h4 class="metrics-title">Analysis</h4>
                <div id="analysis-metrics" class="metrics-data"></div>
            `;
            metricsContainer.appendChild(analysisSection);
        } catch (error) {
            console.error("Error creating analysis metrics section:", error);
            return false;
        }
    }
    
    // Final verification that our elements now exist
    const tumorMetrics = document.getElementById('tumor-metrics');
    const analysisMetrics = document.getElementById('analysis-metrics');
    
    if (!tumorMetrics || !analysisMetrics) {
        console.error("Failed to create metrics elements:", {
            tumorMetrics: !!tumorMetrics,
            analysisMetrics: !!analysisMetrics
        });
        return false;
    }
    
    console.log("Successfully ensured all metrics containers exist");
    return true;
} 