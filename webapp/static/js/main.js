// Main JavaScript for UKT Prediction Web Application

document.addEventListener('DOMContentLoaded', function() {
    console.log('UKT Prediction App Initialized');
    
    // Initialize components
    initializeStudyProgramSearch();
    initializeForm();
    initializeScrollToTop();
    initializeHealthCheck();
    initializePageNav();
    setupGlobalAlertClose();
});

// Health Check Indicator
function initializeHealthCheck() {
    fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            console.log('✓ API Status:', data.status);
            const dot = document.getElementById('apiStatusDot');
            const text = document.getElementById('apiStatusText');
            if (dot && text) {
                if (data.status === 'healthy') {
                    dot.classList.remove('status-down');
                    dot.classList.add('status-up');
                    text.textContent = 'Online';
                } else {
                    dot.classList.remove('status-up');
                    dot.classList.add('status-down');
                    text.textContent = 'Degraded';
                }
            }
        })
        .catch(error => {
            console.error('⚠ API Health Check Failed:', error);
            const dot = document.getElementById('apiStatusDot');
            const text = document.getElementById('apiStatusText');
            if (dot && text) {
                dot.classList.remove('status-up');
                dot.classList.add('status-down');
                text.textContent = 'Offline';
            }
            showAlert('API is not reachable right now. Some features may not work.', 'warning');
        });
}

// Scroll to Top Button
function initializeScrollToTop() {
    const scrollBtn = document.getElementById('scrollToTop');
    if (!scrollBtn) return;
    
    window.addEventListener('scroll', function() {
        if (window.pageYOffset > 300) {
            scrollBtn.classList.add('visible');
        } else {
            scrollBtn.classList.remove('visible');
        }
    });
    
    scrollBtn.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

// Study Program Search Functionality
function initializeStudyProgramSearch() {
    const searchInput = document.getElementById('study_program_search');
    const selectBox = document.getElementById('study_program');
    
    if (!searchInput || !selectBox) return;
    
    const allOptions = Array.from(selectBox.options);
    
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        
        // Filter options
        selectBox.innerHTML = '';
        
        const filteredOptions = allOptions.filter(option => 
            option.value.toLowerCase().includes(searchTerm)
        );
        
        if (filteredOptions.length === 0) {
            const noResult = document.createElement('option');
            noResult.textContent = 'No matching programs found';
            noResult.disabled = true;
            selectBox.appendChild(noResult);
        } else {
            filteredOptions.forEach(option => {
                selectBox.appendChild(option.cloneNode(true));
            });
        }
    });
    
    // Auto-select when clicking on filtered option
    selectBox.addEventListener('click', function() {
        if (this.value) {
            searchInput.value = this.options[this.selectedIndex].text;
        }
    });
}

// Form Submission
function initializeForm() {
    const form = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = document.getElementById('btnText');
    const btnLoading = document.getElementById('btnLoading');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        predictBtn.disabled = true;
        btnText.style.display = 'none';
        btnLoading.style.display = 'inline-block';
        
        // Gather form data
        const formData = {
            Universitas: document.getElementById('university').value,
            Program: document.getElementById('program').value,
            Tahun: document.getElementById('year').value,
            Penerimaan: document.getElementById('admission').value,
            Program_Studi: document.getElementById('study_program').value
        };
        
        try {
            // Make prediction request
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Prediction failed');
            }
            
            const result = await response.json();
            
            if (result.success) {
                displayResults(result, formData);
            } else {
                throw new Error(result.error || 'Unknown error');
            }
            
        } catch (error) {
            console.error('Error:', error);
            showAlert(`Error: ${error.message}. Please check your input and try again.`, 'error');
        } finally {
            // Reset button state
            predictBtn.disabled = false;
            btnText.style.display = 'inline';
            btnLoading.style.display = 'none';
        }
    });
}

// Display Results
function displayResults(result, formData) {
    const resultsCard = document.getElementById('resultsCard');
    const placeholderCard = document.getElementById('placeholderCard');
    const inputSummary = document.getElementById('inputSummary');
    const resultsGrid = document.getElementById('predictionResults');
    
    // Hide placeholder, show results
    placeholderCard.style.display = 'none';
    resultsCard.style.display = 'block';
    
    // Display input summary
    inputSummary.innerHTML = `
        <h4>📝 Input Summary</h4>
        <p><strong>University:</strong> ${formData.Universitas}</p>
        <p><strong>Program:</strong> ${formData.Program}</p>
        <p><strong>Year:</strong> ${formData.Tahun}</p>
        <p><strong>Admission:</strong> ${formData.Penerimaan}</p>
        <p><strong>Study Program:</strong> ${formData.Program_Studi}</p>
    `;
    
    // Display prediction results
    resultsGrid.innerHTML = '';
    
    const predictions = result.predictions;
    // Sort tiers numerically (UKT-1, UKT-2, ..., UKT-11)
    const tiers = Object.keys(predictions).sort((a, b) => {
        const numA = parseInt(a.split('-')[1]);
        const numB = parseInt(b.split('-')[1]);
        return numA - numB;
    });
    
    tiers.forEach(tier => {
        const pred = predictions[tier];
        
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        
        let intervalHTML = '';
        if (pred.lower !== null && pred.upper !== null) {
            intervalHTML = `
                <div class="result-interval">
                    <strong>90% Interval:</strong><br>
                    ${pred.lower_formatted} - ${pred.upper_formatted}
                </div>
            `;
        }
        
        resultItem.innerHTML = `
            <div class="result-tier">${tier}</div>
            <div class="result-value">${pred.formatted}</div>
            ${intervalHTML}
        `;
        
        resultsGrid.appendChild(resultItem);
    });
    
    // Create visualization
    createPredictionChart(predictions, result.has_intervals);
    
    // Smooth scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Create Chart
let chartInstance = null;

function createPredictionChart(predictions, hasIntervals) {
    const ctx = document.getElementById('predictionChart');
    
    // Destroy existing chart
    if (chartInstance) {
        chartInstance.destroy();
    }
    
    // Sort tiers numerically (UKT-1, UKT-2, ..., UKT-11)
    const tiers = Object.keys(predictions).sort((a, b) => {
        const numA = parseInt(a.split('-')[1]);
        const numB = parseInt(b.split('-')[1]);
        return numA - numB;
    });
    const values = tiers.map(tier => predictions[tier].value);
    const lowerBounds = hasIntervals ? tiers.map(tier => predictions[tier].lower) : null;
    const upperBounds = hasIntervals ? tiers.map(tier => predictions[tier].upper) : null;
    
    const datasets = [{
        label: 'Predicted Fee',
        data: values,
        backgroundColor: 'rgba(37, 99, 235, 0.6)',
        borderColor: 'rgba(37, 99, 235, 1)',
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6
    }];
    
    if (hasIntervals) {
        datasets.push({
            label: 'Lower Bound (90% CI)',
            data: lowerBounds,
            backgroundColor: 'rgba(16, 185, 129, 0.2)',
            borderColor: 'rgba(16, 185, 129, 0.8)',
            borderWidth: 1,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false
        });
        
        datasets.push({
            label: 'Upper Bound (90% CI)',
            data: upperBounds,
            backgroundColor: 'rgba(239, 68, 68, 0.2)',
            borderColor: 'rgba(239, 68, 68, 0.8)',
            borderWidth: 1,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: '-1'
        });
    }
    
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: tiers,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2,
            plugins: {
                title: {
                    display: true,
                    text: 'UKT Fee Predictions Across All Tiers',
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    padding: 20
                },
                legend: {
                    display: true,
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += 'Rp ' + context.parsed.y.toLocaleString('id-ID');
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return 'Rp ' + (value / 1000000).toFixed(1) + 'M';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Tuition Fee (IDR)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'UKT Tier'
                    }
                }
            }
        }
    });
}

// API Health Check (for debugging)
async function checkAPIHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API Health Check Failed:', error);
    }
}

// Run health check on load
checkAPIHealth();

// Page navigation (smooth scroll + scrollspy)
function initializePageNav() {
    const nav = document.querySelector('.page-nav');
    if (!nav) return;

    const links = Array.from(nav.querySelectorAll('a[href^="#"]'));
    const ids = links.map(a => a.getAttribute('href').slice(1));
    const sections = ids
        .map(id => document.getElementById(id))
        .filter(Boolean);

    // Smooth scroll with offset for sticky navbar
    links.forEach(link => {
        link.addEventListener('click', (e) => {
            const hash = link.getAttribute('href');
            if (!hash || hash === '#') return;
            const target = document.querySelector(hash);
            if (!target) return;
            e.preventDefault();
            const offset = 80; // approximate navbar height
            const top = target.getBoundingClientRect().top + window.scrollY - offset;
            window.scrollTo({ top, behavior: 'smooth' });
            history.replaceState(null, '', hash);
        });
    });

    // Scrollspy using IntersectionObserver
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const id = entry.target.getAttribute('id');
            if (entry.isIntersecting) {
                links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + id));
            }
        });
    }, { root: null, rootMargin: '-45% 0px -45% 0px', threshold: 0.01 });

    sections.forEach(sec => observer.observe(sec));

    // If page loaded with hash, adjust scroll position
    if (window.location.hash) {
        const target = document.querySelector(window.location.hash);
        if (target) {
            const offset = 80;
            const top = target.getBoundingClientRect().top + window.scrollY - offset;
            window.scrollTo({ top });
        }
    }
}

// Global alert banner utilities
function setupGlobalAlertClose() {
    // No close button by default; alerts auto-hide after 6s
}

function showAlert(message, type = 'info') {
    const banner = document.getElementById('globalAlert');
    if (!banner) return;
    banner.textContent = message;
    banner.className = `alert-banner ${type}`;
    banner.style.display = 'block';
    window.clearTimeout(showAlert._timer);
    showAlert._timer = window.setTimeout(() => {
        banner.style.display = 'none';
    }, 6000);
    banner.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
