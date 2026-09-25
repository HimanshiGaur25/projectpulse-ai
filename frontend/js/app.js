let activeSection = "overview";
let slackFile = null;
const $ = (selector) => document.querySelector(selector);

let uploadedFile = null;
let uploadedRows = [];
let originalContent = "";

// ============================================
// 1. CHECK BACKEND CONNECTION
// ============================================

async function checkBackend() {
    const status = $(".sidebar-footer");

    try {
        const response = await fetch("/api/health");

        if (!response.ok) {
            throw new Error("Backend health check failed");
        }

        const data = await response.json();

        console.log("ProjectPulse AI:", data);

        if (status) {
            status.innerHTML =
                '<span class="status-dot"></span> AI Engine Connected';
        }

    } catch (error) {
        console.error("Backend connection failed:", error);

        if (status) {
            status.textContent = "Backend Disconnected";
        }
    }
}


// ============================================
// 2. NAVIGATION
// ============================================

function navigateTo(section) {
    activeSection = section.toLowerCase();
    const navItems = document.querySelectorAll(".nav-item");

    navItems.forEach((item) => {
        item.classList.remove("active");
    });

    const selectedItem = [...navItems].find((item) =>
        item.textContent.trim().toLowerCase().includes(section.toLowerCase())
    );

    if (selectedItem) {
        selectedItem.classList.add("active");
    }

    const title = $("h1");
    const subtitle = $(".topbar .subtitle");
    const contentGrid = $(".content-grid");
    const statsGrid = $(".stats-grid");

    const sections = {
        overview: {
            title: "Project Overview",
            subtitle: "Monitor project health, team signals and risks.",
            showStats: true
        },

        projects: {
            title: "Projects",
            subtitle: "Review your project portfolio and health.",
            showStats: true,
            content: `
                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <h2>Project Portfolio</h2>
                            <p>Your uploaded project data</p>
                        </div>
                    </div>

                    <div class="empty-state">
                        <div class="empty-icon">◈</div>

                        <h3>
                            ${uploadedFile
                                ? "Project data uploaded"
                                : "No project data yet"}
                        </h3>

                        <p>
                            ${uploadedFile
                                ? escapeHTML(uploadedFile.name) +
                                  " — " + uploadedRows.length +
                                  " data rows"
                                : "Upload a project CSV to view your portfolio."}
                        </p>

                        <button class="primary-button" id="section-upload">
                            Upload CSV
                        </button>

                        ${uploadedFile ? `
                            <button class="primary-button" id="section-analyze">
                                Analyze
                            </button>
                        ` : ""}
                    </div>
                </div>
            `
        },

        "slack insights": {
            title: "Slack Insights",
            subtitle: "Explore communication patterns and project signals.",
            showStats: false,
            content: `
                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <h2>Slack Message Analysis</h2>
                            <p>Analyze communication data.</p>
                        </div>
                    </div>

                    <div class="empty-state">
                        <div class="empty-icon">◉</div>

                        <h3>Ready for Slack data</h3>

                        <p>
                            Upload a Slack CSV to prepare your messages
                            for analysis.
                        </p>

                        <button class="secondary-button" id="slack-upload">
                            Upload Slack CSV
                        </button>

                        <p id="slack-status"></p>
                    </div>
                </div>
            `
        },

        "emotion analysis": {
            title: "Emotion Analysis",
            subtitle: "Explore emotion predictions from text.",
            showStats: false,
            content: `
                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <h2>Emotion Analysis</h2>
                            <p>Analyze text with your emotion model.</p>
                        </div>
                    </div>

                    <div class="empty-state">
                        <div class="empty-icon">✦</div>

                        <h3>Analyze a message</h3>

                        <p>
                            Enter a message below to prepare an
                            emotion prediction.
                        </p>

                        <textarea
                            id="emotion-text"
                            placeholder="Type or paste a message..."
                            rows="5"
                            style="width:100%;box-sizing:border-box;"
                        ></textarea>

                        <button
                            class="primary-button"
                            id="emotion-analyze"
                        >
                            Analyze text
                        </button>

                        <p id="emotion-result"></p>
                    </div>
                </div>
            `
        }
    };

    const page = sections[section] || sections.overview;

    title.textContent = page.title;
    subtitle.textContent = page.subtitle;
    statsGrid.style.display = page.showStats ? "" : "none";

    if (section === "overview") {
        contentGrid.style.display = "";
        contentGrid.innerHTML = originalContent;
        attachOverviewEvents();
    } else {
        contentGrid.style.display = "block";
        contentGrid.innerHTML = page.content;
        attachSectionEvents(section);
    }
}


// ============================================
// 3. SAFE HTML TEXT
// ============================================

function escapeHTML(value) {
    return String(value).replace(/[&<>"']/g, (character) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
    })[character]);
}


// ============================================
// 4. CSV FILE PICKER
// ============================================

function openCSVPicker() {
    let picker = $("#csv-file-picker");

    if (!picker) {
        picker = document.createElement("input");

        picker.type = "file";
        picker.id = "csv-file-picker";
        picker.accept = ".csv,text/csv";
        picker.hidden = true;

        document.body.appendChild(picker);

        picker.addEventListener("change", handleFileUpload);
    }

    // Allow selecting the same file again
    picker.value = "";
    picker.click();
}


// ============================================
// 5. READ UPLOADED CSV
// ============================================

async function handleFileUpload(event) {
    const file = event.target.files[0];

    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".csv")) {
        alert("Please select a CSV file.");
        return;
    }

    uploadedFile = file;

    try {
        const csvText = await file.text();

        const lines = csvText
            .split(/\r?\n/)
            .filter((line) => line.trim());

        uploadedRows = lines.length > 1
            ? lines.slice(1)
            : [];

        const emptyState = $(".empty-state");

        if (emptyState) {
            emptyState.innerHTML = `
                <div class="empty-icon">✓</div>

                <h3>CSV uploaded successfully</h3>

                <p>${escapeHTML(file.name)}</p>

                <p>${uploadedRows.length} data rows detected.</p>

                <button
                    class="primary-button"
                    id="analyze-upload"
                >
                    Analyze
                </button>
            `;

            $("#analyze-upload").addEventListener(
                "click",
                analyzeData
            );
        }

        // Update the row count temporarily
        const total = $("#total-projects");

        if (total) {
            total.textContent = uploadedRows.length;
        }

        alert(
            `File uploaded: ${file.name}\n` +
            `${uploadedRows.length} data rows detected.`
        );

    } catch (error) {
        console.error("Could not read CSV:", error);

        alert("Unable to read this file. Please try another CSV.");
    }
}


// ============================================
// 6. ANALYZE PROJECT CSV USING FASTAPI
// ============================================

async function analyzeData() {
    if (!uploadedFile) {
        alert("Please upload a project CSV first.");
        return;
    }

    const analyzeButton = $("#analyze-upload");

    if (analyzeButton) {
        analyzeButton.disabled = true;
        analyzeButton.textContent = "Analyzing...";
    }

    try {
        // Prepare the CSV for the Python backend
        const formData = new FormData();

        formData.append("file", uploadedFile);

        // Send CSV to the real analysis endpoint
        const response = await fetch("/api/projects/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || `Request failed: ${response.status}`
            );
        }

        if (!data.summary || !Array.isArray(data.projects)) {
            throw new Error("The backend returned an unexpected response.");
        }

        // Update KPI cards
        $("#total-projects").textContent = data.summary.total;
        $("#healthy-projects").textContent = data.summary.healthy;
        $("#risk-projects").textContent = data.summary.at_risk;
        $("#critical-projects").textContent = data.summary.critical;

        // Show project results
        const contentGrid = $(".content-grid");

        contentGrid.style.display = "block";

        contentGrid.innerHTML = `
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <h2>Project Health Results</h2>
                        <p>
                            Analysis of ${escapeHTML(data.filename)}
                        </p>
                    </div>

                    <button
                        class="secondary-button"
                        id="upload-another"
                    >
                        Upload Another CSV
                    </button>
                </div>

                <div style="overflow-x:auto;">
                    <table style="
                        width:100%;
                        border-collapse:collapse;
                        text-align:left;
                    ">
                        <thead>
                            <tr>
                                <th style="padding:12px;">Project</th>
                                <th style="padding:12px;">Client</th>
                                <th style="padding:12px;">Owner</th>
                                <th style="padding:12px;">Score</th>
                                <th style="padding:12px;">Status</th>
                                <th style="padding:12px;">Risk Signals</th>
                            </tr>
                        </thead>

                        <tbody>
                            ${data.projects.map((project) => `
                                <tr>
                                    <td style="padding:12px;">
                                        ${escapeHTML(project.project_name)}
                                    </td>

                                    <td style="padding:12px;">
                                        ${escapeHTML(project.client)}
                                    </td>

                                    <td style="padding:12px;">
                                        ${escapeHTML(project.owner)}
                                    </td>

                                    <td style="padding:12px;">
                                        ${project.health_score}/100
                                    </td>

                                    <td style="padding:12px;">
                                        <strong>
                                            ${escapeHTML(project.health_label)}
                                        </strong>
                                    </td>

                                    <td style="padding:12px;">
                                        ${escapeHTML(project.risk_reason)}
                                    </td>
                                </tr>
                            `).join("")}
                        </tbody>
                    </table>
                </div>
            </div>
        `;

        $("#upload-another").addEventListener(
            "click",
            openCSVPicker
        );

        console.log("Project analysis results:", data);

    } catch (error) {
        console.error("Project analysis failed:", error);

        alert("Analysis failed: " + error.message);

    } finally {
        const currentButton = $("#analyze-upload");

        if (currentButton) {
            currentButton.disabled = false;
            currentButton.textContent = "Analyze";
        }
    }
}


// ============================================
// 7. OVERVIEW BUTTONS
// ============================================

function attachOverviewEvents() {
    document.querySelectorAll(".topbar-actions button").forEach(
        (button) => {
            const text = button.textContent.trim().toLowerCase();

            if (text.includes("upload")) {
                button.onclick = () => {
                    if (activeSection === "slack insights") {
                        openSlackCSVPicker();
                    } else {
                        openCSVPicker();
                    }
                };
            }

            if (text === "analyze") {
                button.onclick = () => {
                    if (activeSection === "slack insights") {
                        analyzeSlackMessages();
                    } else {
                        analyzeData();
                    }
                };
            }
        }
    );

    document.querySelectorAll(".content-grid button").forEach(
        (button) => {
            const text = button.textContent.trim().toLowerCase();

            if (text.includes("upload")) {
                button.onclick = () => {
                    if (activeSection === "slack insights") {
                        openSlackCSVPicker();
                    } else {
                        openCSVPicker();
                    }
                };
            }

            if (text === "analyze") {
                button.onclick = () => {
                    if (activeSection === "slack insights") {
                        analyzeSlackMessages();
                    } else {
                        analyzeData();
                    }
                };
            }

            if (text === "view all") {
                button.onclick = () => navigateTo("projects");
            }
        }
    );
}


// ============================================
// 8. OTHER SECTION BUTTONS
// ============================================

function attachSectionEvents(section) {
    const slackUpload = document.querySelector("#slack-upload");

    if (slackUpload) {
        slackUpload.onclick = openSlackCSVPicker;
    }

    const slackAnalyze = document.querySelector("#slack-analyze");

    if (slackAnalyze) {
        slackAnalyze.onclick = analyzeSlackMessages;
    }

    const emotionButton = $("#emotion-analyze");

    if (emotionButton) {
        emotionButton.addEventListener("click", () => {
            const text = $("#emotion-text").value.trim();

            if (!text) {
                alert("Please enter some text first.");
                return;
            }

            $("#emotion-result").textContent =
                "Text entered. The emotion prediction endpoint " +
                "still needs to be connected.";
        });
    }
}

/* ============================================
   SLACK INSIGHTS
============================================ */

function openSlackCSVPicker() {
    let picker = document.querySelector("#slack-file-picker");

    if (!picker) {
        picker = document.createElement("input");
        picker.type = "file";
        picker.id = "slack-file-picker";
        picker.accept = ".csv,text/csv";
        picker.hidden = true;

        document.body.appendChild(picker);

        picker.addEventListener("change", handleSlackFileUpload);
    }

    picker.value = "";
    picker.click();
}


async function handleSlackFileUpload(event) {
    const file = event.target.files[0];

    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".csv")) {
        alert("Please select a CSV file.");
        return;
    }

    slackFile = file;

    const status = document.querySelector("#slack-status");

    if (status) {
        status.textContent = `${file.name} selected.`;
    }

    // Show the Analyze button on the Slack page
    const uploadButton = document.querySelector("#slack-upload");

    if (uploadButton) {
        uploadButton.textContent = "Change CSV";
    }

    let analyzeButton = document.querySelector("#slack-analyze");

    if (!analyzeButton) {
        analyzeButton = document.createElement("button");
        analyzeButton.id = "slack-analyze";
        analyzeButton.className = "primary-button";
        analyzeButton.textContent = "Analyze Messages";

        if (uploadButton) {
            uploadButton.insertAdjacentElement(
                "afterend",
                analyzeButton
            );
        } else if (status) {
            status.insertAdjacentElement(
                "afterend",
                analyzeButton
            );
        }

        analyzeButton.addEventListener(
            "click",
            analyzeSlackMessages
        );
    }
}


async function analyzeSlackMessages() {
    if (!slackFile) {
        alert("Please upload a Slack CSV first.");
        return;
    }

    const button = document.querySelector("#slack-analyze");
    const status = document.querySelector("#slack-status");

    if (button) {
        button.disabled = true;
        button.textContent = "Analyzing...";
    }

    if (status) {
        status.textContent = "Analyzing Slack messages...";
    }

    try {
        const formData = new FormData();
        formData.append("file", slackFile);

        const response = await fetch("/api/slack/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || `Request failed: ${response.status}`
            );
        }

        renderSlackResults(data);

        if (status) {
            status.textContent =
                `Analyzed ${data.summary.total_messages} messages.`;
        }

    } catch (error) {
        console.error("Slack analysis failed:", error);

        if (status) {
            status.textContent = "Analysis failed: " + error.message;
        } else {
            alert("Slack analysis failed: " + error.message);
        }

    } finally {
        const currentButton =
            document.querySelector("#slack-analyze");

        if (currentButton) {
            currentButton.disabled = false;
            currentButton.textContent = "Analyze Messages";
        }
    }
}


function renderSlackResults(data) {
    const contentGrid = document.querySelector(".content-grid");

    if (!contentGrid) return;

    const counts = data.summary.category_counts || {};

    const categoryCards = Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .map(([category, count]) => `
            <div class="panel" style="padding:18px;">
                <p style="margin:0 0 8px;color:var(--text-secondary,#777);">
                    ${escapeHTML(category)}
                </p>

                <h2 style="margin:0;">
                    ${count}
                </h2>
            </div>
        `).join("");

    const messageRows = data.messages.map((item) => `
        <tr>
            <td style="padding:12px;min-width:250px;">
                ${escapeHTML(item.message)}
            </td>

            <td style="padding:12px;">
                <span class="status-badge">
                    ${escapeHTML(item.category)}
                </span>
            </td>

            <td style="padding:12px;">
                ${item.confidence == null
                    ? "—"
                    : `${item.confidence}%`}
            </td>

            <td style="padding:12px;">
                ${escapeHTML(
                    item.channel ||
                    item.username ||
                    item.user ||
                    "—"
                )}
            </td>

            <td style="padding:12px;">
                ${escapeHTML(
                    item.date ||
                    item.timestamp ||
                    "—"
                )}
            </td>
        </tr>
    `).join("");

    contentGrid.style.display = "block";

    contentGrid.innerHTML = `
        <div class="panel">
            <div class="panel-header">
                <div>
                    <h2>Slack Insights</h2>
                    <p>
                        ${escapeHTML(data.filename)} ·
                        ${data.summary.total_messages} messages analyzed
                    </p>
                </div>

                <button
                    class="secondary-button"
                    id="slack-upload-another"
                >
                    Upload Another CSV
                </button>
            </div>
        </div>

        <div class="stats-grid" style="margin:20px 0;">
            <div class="stat-card">
                <p>Total Messages</p>
                <h2>${data.summary.total_messages}</h2>
            </div>

            <div class="stat-card">
                <p>Categories Found</p>
                <h2>${data.summary.categories}</h2>
            </div>
        </div>

        <h2 style="margin:24px 0 14px;">
            Message Categories
        </h2>

        <div style="
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
            gap:16px;
        ">
            ${categoryCards}
        </div>

        <div class="panel" style="margin-top:24px;">
            <div class="panel-header">
                <div>
                    <h2>Classified Messages</h2>
                    <p>Predictions from your trained Slack ML model</p>
                </div>
            </div>

            <div style="overflow-x:auto;">
                <table style="
                    width:100%;
                    border-collapse:collapse;
                    text-align:left;
                ">
                    <thead>
                        <tr>
                            <th style="padding:12px;">Message</th>
                            <th style="padding:12px;">Category</th>
                            <th style="padding:12px;">Confidence</th>
                            <th style="padding:12px;">User / Channel</th>
                            <th style="padding:12px;">Date</th>
                        </tr>
                    </thead>

                    <tbody>
                        ${messageRows}
                    </tbody>
                </table>
            </div>
        </div>
    `;

    document.querySelector("#slack-upload-another")
        .addEventListener("click", openSlackCSVPicker);
}

// ============================================
// 9. INITIALIZE APP
// ============================================

document.addEventListener("DOMContentLoaded", () => {
    const contentGrid = $(".content-grid");

    if (contentGrid) {
        originalContent = contentGrid.innerHTML;
    }

    // Sidebar navigation
    document.querySelectorAll(".nav-item").forEach((item) => {
        item.addEventListener("click", () => {
            const text = item.textContent.trim().toLowerCase();

            if (text.includes("overview")) {
                navigateTo("overview");
            } else if (text.includes("projects")) {
                navigateTo("projects");
            } else if (text.includes("slack")) {
                navigateTo("slack insights");
            } else if (text.includes("emotion")) {
                navigateTo("emotion analysis");
            }
        });
    });

    attachOverviewEvents();
    checkBackend();
});