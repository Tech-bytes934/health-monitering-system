document.addEventListener("DOMContentLoaded", function () {
    // Switching between login & signup forms
    const loginForm = document.getElementById("login-form");
    const signupForm = document.getElementById("signup-form");
    const showSignup = document.getElementById("show-signup");
    const showLogin = document.getElementById("show-login");

    if (showSignup && showLogin) {
        showSignup.addEventListener("click", function (e) {
            e.preventDefault();
            loginForm.classList.add("hidden");
            signupForm.classList.remove("hidden");
        });

        showLogin.addEventListener("click", function (e) {
            e.preventDefault();
            signupForm.classList.add("hidden");
            loginForm.classList.remove("hidden");
        });
    }

    // Login Form Validation
    if (loginForm) {
        loginForm.addEventListener("submit", function (e) {
            e.preventDefault();
            const email = document.getElementById("login-email").value;
            const password = document.getElementById("login-password").value;

            if (email === "test@example.com" && password === "password") {
                alert("Login successful!");
                window.location.href = "dashboard.html";
            } else {
                alert("Invalid credentials!");
            }
        });
    }

    // Signup Form Validation
    if (signupForm) {
        signupForm.addEventListener("submit", function (e) {
            e.preventDefault();
            const name = document.getElementById("signup-name").value;
            alert(`Account created for ${name}!`);
            loginForm.classList.remove("hidden");
            signupForm.classList.add("hidden");
        });
    }

    // Feature Card Animation on Scroll
    const featureCards = document.querySelectorAll(".feature-card");
    function checkVisibility() {
        featureCards.forEach((card) => {
            if (card.getBoundingClientRect().top < window.innerHeight - 50) {
                card.classList.add("active");
            }
        });
    }
    window.addEventListener("scroll", checkVisibility);
    checkVisibility(); // Run on page load

    // Feature Menu Toggle Behavior
    const toggleButton = document.querySelector('[data-bs-target="#featureMenu"]');
    const featureMenu = document.querySelector("#featureMenu");

    if (toggleButton && featureMenu) {
        document.addEventListener("click", function (event) {
            if (!featureMenu.contains(event.target) && !toggleButton.contains(event.target)) {
                featureMenu.classList.remove("show");
            }
        });

        document.querySelectorAll("#featureMenu a").forEach(link => {
            link.addEventListener("click", function () {
                featureMenu.classList.remove("show");
            });
        });
    }

    // AI-Powered Health Monitoring (Vitals + SOS)
    const sosButton = document.getElementById("sos-button");
    const alertMessage = document.getElementById("sos-alert");
    const heartRate = document.getElementById("heart-rate");
    const bp = document.getElementById("bp");
    const locationBox = document.getElementById("location");

    function updateVitals() {
        heartRate.textContent = `${Math.floor(Math.random() * (120 - 60) + 60)} BPM`;
        bp.textContent = `${Math.floor(Math.random() * (140 - 90) + 90)}/${Math.floor(Math.random() * (90 - 60) + 60)} mmHg`;
    }

    if (sosButton) {
        sosButton.addEventListener("click", function () {
            alertMessage.classList.remove("d-none");
            sosButton.classList.add("disabled");
        });
    }

    function getLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                function (position) {
                    locationBox.textContent = `Lat: ${position.coords.latitude}, Lon: ${position.coords.longitude}`;
                },
                function () {
                    locationBox.textContent = "Location access denied!";
                }
            );
        } else {
            locationBox.textContent = "Geolocation not supported!";
        }
    }

    if (locationBox) getLocation();
    setInterval(updateVitals, 5000); // Update vitals every 5 seconds

    // AI Period Tracker
    const predictPeriodBtn = document.getElementById("predict-period");
    if (predictPeriodBtn) {
        predictPeriodBtn.addEventListener("click", function () {
            let lastPeriod = document.getElementById("last-period").value;
            let cycleLength = parseInt(document.getElementById("cycle-length").value);

            if (!lastPeriod || isNaN(cycleLength)) {
                alert("Please enter a valid period start date and cycle length.");
                return;
            }

            let nextPeriodDate = new Date(lastPeriod);
            nextPeriodDate.setDate(nextPeriodDate.getDate() + cycleLength);

            document.getElementById("modal-period-result").innerHTML =
                `Your next period is expected on: <strong>${nextPeriodDate.toISOString().split('T')[0]}</strong>`;
        });
    }

    // AI Fertility Prediction
    const fertilityCheckBtn = document.getElementById("fertility-check");
    if (fertilityCheckBtn) {
        fertilityCheckBtn.addEventListener("click", () => {
            document.getElementById("fertility-result").textContent =
                "Based on AI predictions, your best conception days are in the next 3-5 days.";
            document.getElementById("fertility-result").classList.remove("d-none");
        });
    }

    // AI Hormonal Imbalance Checker
    const hormonalCheckBtn = document.getElementById("hormonal-check");
    if (hormonalCheckBtn) {
        hormonalCheckBtn.addEventListener("click", () => {
            let selectedSymptoms = [...document.querySelectorAll(".form-check-input:checked")].map(cb => cb.value);

            if (selectedSymptoms.length === 0) {
                alert("Please select at least one symptom.");
                return;
            }

            document.getElementById("hormonal-result").textContent =
                `Based on AI analysis, you may have hormonal imbalance related to: ${selectedSymptoms.join(", ")}`;
            document.getElementById("hormonal-result").classList.remove("d-none");
        });
    }
});
document.getElementById("fertility-check").addEventListener("click", function () {
    let ovulationDate = document.getElementById("ovulation-date").value;

    if (!ovulationDate) {
        alert("Please enter your ovulation date.");
        return;
    }

    let ovulationDay = new Date(ovulationDate);
    let fertileWindowStart = new Date(ovulationDay);
    let fertileWindowEnd = new Date(ovulationDay);

    // Fertile window is generally 5 days before ovulation and the day of ovulation
    fertileWindowStart.setDate(fertileWindowStart.getDate() - 5);
    fertileWindowEnd.setDate(fertileWindowEnd.getDate());

    let formattedStart = fertileWindowStart.toISOString().split('T')[0];
    let formattedEnd = fertileWindowEnd.toISOString().split('T')[0];

    document.getElementById("fertility-result").innerHTML = `
        <strong>AI Prediction:</strong> Your most fertile days are between 
        <strong class="text-danger">${formattedStart}</strong> and <strong class="text-danger">${formattedEnd}</strong>.
        This is the best time for conception.
    `;

    document.getElementById("fertility-result").classList.remove("d-none");
});
document.getElementById("dark-mode-icon").addEventListener("click", function () {
    document.body.classList.toggle("dark-mode");
});

// Chatbot Functionality
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

const responses = {
    "hi": "Hello! How can I assist you today? 😊",
    "menstrual cycle": "The average cycle is 28 days. Do you need tracking tips? 📅",
    "period symptoms": "Common symptoms: cramps, bloating, mood swings. Want remedies? 💡",
    "pcos": "PCOS is a hormonal imbalance. Symptoms: irregular periods, acne. Need guidance? 👩‍⚕️",
    "pregnancy": "Pregnancy care: Nutrition, exercise, and prenatal vitamins. Any specific questions? 🤰",
    "mental health": "Meditation, journaling, and deep breathing help. Need self-care tips? 🧘‍♀️",
    "thank you": "You're welcome! Stay healthy! 🌸"
};

sendBtn.addEventListener("click", sendMessage);
chatInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") sendMessage();
});

function sendMessage() {
    const userText = chatInput.value.trim().toLowerCase();
    if (userText === "") return;

    appendMessage(userText, "user-message");
    chatInput.value = "";

    setTimeout(() => {
        appendMessage(responses[userText] || "I'm still learning! Ask me another question. 🤖", "bot-message");
    }, 500);
}

function appendMessage(text, className) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add(className);
    messageDiv.textContent = text;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
