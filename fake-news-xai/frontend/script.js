/* =========================================================
   TRUTHLENS - JAVASCRIPT
   ========================================================= */


/* =========================================================
   1. LOGIN
   ========================================================= */

const loginForm = document.getElementById("loginForm");

if (loginForm) {

    loginForm.addEventListener("submit", function (event) {

        event.preventDefault();

        const email = document.getElementById("email").value;
        const password = document.getElementById("password").value;

        if (email === "" || password === "") {
            alert("Please enter your email and password.");
            return;
        }

        // Store login status
        localStorage.setItem("truthLensLoggedIn", "true");

        // Go to dashboard
        window.location.href = "dashboard.html";
    });
}


/* =========================================================
   2. FILE SELECTION
   ========================================================= */

function selectFile(inputId) {

    const input = document.getElementById(inputId);

    if (!input) {
        return;
    }

    input.click();

    input.onchange = function () {

        if (input.files.length > 0) {

            const file = input.files[0];

            console.log("Selected file:", file.name);

            analyzeFile(file, inputId);
        }
    };
}


/* =========================================================
   3. AUDIO / VIDEO ANALYSIS (real backend)
   ========================================================= */

function analyzeFile(file, inputId) {

    const fileType =
        inputId === "audioInput"
            ? "Audio"
            : "Video";

    transcribeAndAnalyze(file, fileType);
}


async function transcribeAndAnalyze(file, type) {

    const formData = new FormData();
    formData.append("file", file);

    try {

        const transcribeRes = await fetch("/api/transcribe", {
            method: "POST",
            body: formData
        });

        const transcribeData = await transcribeRes.json();

        if (!transcribeRes.ok) {
            throw new Error(transcribeData.error || "Transcription failed.");
        }

        const result = await classifyText(transcribeData.text);

        updateReport(file.name, type, result);
        saveHistory(file.name, type.toLowerCase(), result);

    } catch (error) {

        alert(
            `Couldn't fully analyze this ${type.toLowerCase()} file: ${error.message}\n\n` +
            `Showing a placeholder result instead.`
        );

        const fallback = {
            probability: type === "Video" ? 32 : 52,
            status: type === "Video" ? "Likely Real" : "Uncertain"
        };

        updateReport(file.name, type, fallback);
        saveHistory(file.name, type.toLowerCase(), fallback);
    }
}


/* =========================================================
   4. TEXT MODAL
   ========================================================= */

const textModal = document.getElementById("textModal");
const openTextModal = document.getElementById("openTextModal");
const closeModal = document.getElementById("closeModal");


if (openTextModal) {

    openTextModal.addEventListener("click", function () {

        textModal.classList.add("active");

    });
}


if (closeModal) {

    closeModal.addEventListener("click", function () {

        textModal.classList.remove("active");

    });
}


/* Close modal by clicking outside */

if (textModal) {

    textModal.addEventListener("click", function (event) {

        if (event.target === textModal) {

            textModal.classList.remove("active");

        }

    });
}


/* =========================================================
   5. TEXT ANALYSIS
   ========================================================= */

const analyzeTextButton =
    document.getElementById("analyzeText");


if (analyzeTextButton) {

    analyzeTextButton.addEventListener("click", async function () {

        const newsText =
            document.getElementById("newsText").value.trim();


        if (newsText === "") {

            alert("Please enter some news or a claim first.");

            return;
        }

        analyzeTextButton.disabled = true;
        analyzeTextButton.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...`;

        const result = await classifyText(newsText);

        analyzeTextButton.disabled = false;
        analyzeTextButton.innerHTML = `<i class="fa-solid fa-magnifying-glass"></i> Analyze with AI`;

        textModal.classList.remove("active");


        updateReport(
            newsText.substring(0, 55) +
            (newsText.length > 55 ? "..." : ""),
            "Text",
            result
        );


        saveHistory(
            newsText.substring(0, 60),
            "text",
            result
        );
    });
}


/* =========================================================
   6. TEXT CLASSIFICATION (calls the real trained model)
   ========================================================= */

async function classifyText(text) {

    try {

        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text, num_features: 8 })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Analysis failed.");
        }

        const probability = Math.round(data.probs.FAKE * 100);

        let status;

        if (probability >= 65) {

            status = "Likely Fake";

        } else if (probability >= 45) {

            status = "Uncertain";

        } else {

            status = "Likely Real";

        }

        return {

            probability: probability,

            status: status,

            wordWeights: data.word_weights

        };

    } catch (error) {

        console.warn(
            "Real model unavailable, falling back to offline demo scoring:",
            error.message
        );

        return demoAnalyzeText(text);
    }
}


/* =========================================================
   6b. OFFLINE FALLBACK
   Used only if the real backend/model can't be reached
   (e.g. server not running, or no trained model yet).
   This is NOT a real AI model -- just keyword counting.
   ========================================================= */

function demoAnalyzeText(text) {

    const lowerText = text.toLowerCase();

    const suspiciousWords = [

        "breaking",
        "shocking",
        "urgent",
        "100% true",
        "must share",
        "viral",
        "secret",
        "government is hiding",
        "miracle",
        "guaranteed",
        "you won't believe",
        "forward this"

    ];


    let suspiciousCount = 0;


    suspiciousWords.forEach(function (word) {

        if (lowerText.includes(word)) {

            suspiciousCount++;

        }

    });


    let probability = 25;


    probability += suspiciousCount * 9;


    if (text.length < 80) {

        probability += 8;

    }


    probability = Math.min(probability, 95);


    let status;

    if (probability >= 65) {

        status = "Likely Fake";

    } else if (probability >= 45) {

        status = "Uncertain";

    } else {

        status = "Likely Real";

    }


    return {

        probability: probability,

        status: status,

        suspiciousCount: suspiciousCount

    };
}


/* =========================================================
   8. UPDATE ANALYSIS REPORT
   ========================================================= */

function updateReport(title, type, result) {

    const reportTitle =
        document.getElementById("reportTitle");

    const resultStatus =
        document.getElementById("resultStatus");

    const fakePercentage =
        document.getElementById("fakePercentage");

    const resultDescription =
        document.getElementById("resultDescription");

    const summaryText =
        document.getElementById("summaryText");


    if (!reportTitle) {
        return;
    }


    reportTitle.textContent = title;


    /*
        Probability
    */

    fakePercentage.textContent =
        result.probability + "%";


    /*
        Status
    */

    resultStatus.textContent =
        result.status;


    /*
        Change colors depending on result
    */

    if (result.status === "Likely Fake") {

        resultStatus.style.color = "#ff6074";

        fakePercentage.style.color = "#ff6074";

        resultDescription.textContent =
            "This content contains several suspicious signals " +
            "that may indicate misinformation or fabricated claims.";

        summaryText.textContent =
            "The analysis detected potentially misleading patterns. " +
            "The content should be verified using reliable sources " +
            "before being shared.";

    }


    else if (result.status === "Uncertain") {

        resultStatus.style.color = "#ffb43b";

        fakePercentage.style.color = "#ffb43b";

        resultDescription.textContent =
            "Some suspicious signals were detected, but there " +
            "is not enough evidence to confidently classify the content.";

        summaryText.textContent =
            "The content requires additional verification. " +
            "Check the original source, supporting evidence, " +
            "and trusted news organizations.";

    }


    else {

        resultStatus.style.color = "#20d7a6";

        fakePercentage.style.color = "#20d7a6";

        resultDescription.textContent =
            "The content does not show strong misinformation " +
            "signals based on this demo analysis.";

        summaryText.textContent =
            "The content appears relatively credible based on " +
            "the available signals. However, users should still " +
            "verify important claims independently.";

    }


    /*
        Update content type
    */

    const reportBadge =
        document.querySelector(
            ".report-content .type-badge"
        );


    if (reportBadge) {

        reportBadge.textContent = type;

    }


    /*
        Update date
    */

    const reportDate =
        document.getElementById("reportDate");


    if (reportDate) {

        reportDate.textContent =
            new Date().toLocaleString();

    }


    /*
        Update the finding
    */

    updateFindings(result);
}


/* =========================================================
   9. UPDATE FINDINGS
   ========================================================= */

function updateFindings(result) {

    const findings =
        document.querySelector(".findings");


    if (!findings) {
        return;
    }


    const findingItems =
        findings.querySelectorAll(".finding-item");


    if (findingItems.length < 4) {
        return;
    }


    /*
        Fake
    */

    if (result.status === "Likely Fake") {

        findingItems[0]
            .querySelector("h4")
            .textContent = "Suspicious Content";

        findingItems[0]
            .querySelector("p")
            .textContent =
            "The content contains language patterns commonly found in misleading posts.";


        findingItems[1]
            .querySelector("h4")
            .textContent = "Manipulated Language";

        findingItems[1]
            .querySelector("p")
            .textContent =
            "Emotional or sensational wording may influence the reader.";


        findingItems[2]
            .querySelector("h4")
            .textContent = "Needs Verification";

        findingItems[2]
            .querySelector("p")
            .textContent =
            "Additional supporting evidence should be checked.";


        findingItems[3]
            .querySelector("h4")
            .textContent = "Fact Check Recommended";

        findingItems[3]
            .querySelector("p")
            .textContent =
            "Verify the claim using reliable sources.";

    }


    /*
        Real
    */

    else if (result.status === "Likely Real") {

        findingItems[0]
            .querySelector("h4")
            .textContent = "No Major Red Flags";

        findingItems[0]
            .querySelector("p")
            .textContent =
            "No strong misinformation patterns were detected.";


        findingItems[1]
            .querySelector("h4")
            .textContent = "Consistent Language";

        findingItems[1]
            .querySelector("p")
            .textContent =
            "The wording does not show strong sensational patterns.";


        findingItems[2]
            .querySelector("h4")
            .textContent = "Evidence Required";

        findingItems[2]
            .querySelector("p")
            .textContent =
            "Important claims should still be verified.";


        findingItems[3]
            .querySelector("h4")
            .textContent = "Appears Credible";

        findingItems[3]
            .querySelector("p")
            .textContent =
            "The content appears relatively consistent.";

    }


    /*
        Uncertain
    */

    else {

        findingItems[0]
            .querySelector("h4")
            .textContent = "Limited Evidence";

        findingItems[0]
            .querySelector("p")
            .textContent =
            "There is not enough information to establish credibility.";


        findingItems[1]
            .querySelector("h4")
            .textContent = "Some Suspicious Signals";

        findingItems[1]
            .querySelector("p")
            .textContent =
            "Certain patterns require additional investigation.";


        findingItems[2]
            .querySelector("h4")
            .textContent = "Source Verification";

        findingItems[2]
            .querySelector("p")
            .textContent =
            "Check whether the original source is trustworthy.";


        findingItems[3]
            .querySelector("h4")
            .textContent = "Further Checking Needed";

        findingItems[3]
            .querySelector("p")
            .textContent =
            "Do not rely on this result alone.";
    }
}


/* =========================================================
   10. SAVE HISTORY
   ========================================================= */

function saveHistory(title, type, result) {

    let history =
        JSON.parse(
            localStorage.getItem("truthLensHistory")
        ) || [];


    const newItem = {

        title: title,

        type: type,

        probability: result.probability,

        status: result.status,

        date: new Date().toLocaleString()

    };


    history.unshift(newItem);


    /*
        Keep only latest 20 analyses
    */

    history = history.slice(0, 20);


    localStorage.setItem(
        "truthLensHistory",
        JSON.stringify(history)
    );
}


/* =========================================================
   11. DISPLAY HISTORY
   ========================================================= */

const historyList =
    document.getElementById("historyList");


if (historyList) {

    displayHistory();

}


function displayHistory(filter = "all") {

    const list =
        document.getElementById("historyList");


    if (!list) {
        return;
    }


    let history =
        JSON.parse(
            localStorage.getItem("truthLensHistory")
        ) || [];


    /*
        If no user history exists,
        keep the demo history already in HTML.
    */

    if (history.length === 0) {
        return;
    }


    if (filter !== "all") {

        history =
            history.filter(function (item) {

                return item.type === filter;

            });

    }


    list.innerHTML = "";


    if (history.length === 0) {

        list.innerHTML = `
            <div class="empty-history">
                <i class="fa-solid fa-clock"></i>
                <p>No analysis found.</p>
            </div>
        `;

        return;
    }


    history.forEach(function (item) {

        const card =
            document.createElement("div");


        card.className = "history-card";

        card.dataset.type = item.type;


        let icon = "fa-file-lines";

        if (item.type === "audio") {
            icon = "fa-microphone";
        }

        if (item.type === "video") {
            icon = "fa-play";
        }


        let scoreClass = "fake-score";

        if (item.status === "Likely Real") {
            scoreClass = "real-score";
        }

        if (item.status === "Uncertain") {
            scoreClass = "uncertain-score";
        }


        card.innerHTML = `

            <div class="history-type">
                <i class="fa-solid ${icon}"></i>
            </div>

            <div class="history-info">

                <h3>${escapeHTML(item.title)}</h3>

                <p>
                    ${capitalize(item.type)}
                    Analysis • ${item.date}
                </p>

            </div>

            <div class="history-score ${scoreClass}">

                ${item.probability}%

                <span>
                    ${item.status}
                </span>

            </div>

            <button class="view-report-btn">
                View Report
            </button>
        `;


        list.appendChild(card);

    });
}


/* =========================================================
   12. HISTORY FILTERS
   ========================================================= */

const filterButtons =
    document.querySelectorAll(".filter-btn");


filterButtons.forEach(function (button) {

    button.addEventListener("click", function () {

        filterButtons.forEach(function (btn) {

            btn.classList.remove("active");

        });


        button.classList.add("active");


        const filter =
            button.dataset.filter;


        displayHistory(filter);

    });

});


/* =========================================================
   13. CLEAR HISTORY
   ========================================================= */

const clearHistory =
    document.getElementById("clearHistory");


if (clearHistory) {

    clearHistory.addEventListener("click", function () {

        const confirmDelete =
            confirm(
                "Are you sure you want to clear your analysis history?"
            );


        if (!confirmDelete) {
            return;
        }


        localStorage.removeItem(
            "truthLensHistory"
        );


        const list =
            document.getElementById("historyList");


        if (list) {

            list.innerHTML = `

                <div class="empty-history">

                    <i class="fa-solid fa-clock"></i>

                    <p>
                        No analysis history yet.
                    </p>

                </div>
            `;

        }

    });

}


/* =========================================================
   14. CHECK ANOTHER
   ========================================================= */

const checkAnother =
    document.getElementById("checkAnother");


if (checkAnother) {

    checkAnother.addEventListener("click", function () {

        const textModal =
            document.getElementById("textModal");


        if (textModal) {

            textModal.classList.add("active");

        }

    });

}


/* =========================================================
   15. SETTINGS - DARK MODE
   ========================================================= */

const darkModeToggle =
    document.getElementById("darkModeToggle");


if (darkModeToggle) {

    const savedMode =
        localStorage.getItem("truthLensDarkMode");


    if (savedMode === "false") {

        darkModeToggle.checked = false;

    }


    darkModeToggle.addEventListener(
        "change",
        function () {

            localStorage.setItem(
                "truthLensDarkMode",
                darkModeToggle.checked
            );


            if (darkModeToggle.checked) {

                document.body.classList.remove(
                    "light-mode"
                );

            } else {

                document.body.classList.add(
                    "light-mode"
                );

            }

        }
    );
}


/* =========================================================
   16. SAVE SETTINGS
   ========================================================= */

const saveSettings =
    document.getElementById("saveSettings");


if (saveSettings) {

    saveSettings.addEventListener("click", function () {

        saveSettings.innerHTML = `
            <i class="fa-solid fa-check"></i>
            Saved!
        `;


        setTimeout(function () {

            saveSettings.innerHTML = `
                <i class="fa-solid fa-check"></i>
                Save Changes
            `;

        }, 2000);

    });

}


/* =========================================================
   17. HELPER - CAPITALIZE
   ========================================================= */

function capitalize(text) {

    if (!text) {
        return "";
    }

    return text.charAt(0).toUpperCase() +
        text.slice(1);

}


/* =========================================================
   18. HELPER - SECURITY
   ========================================================= */

function escapeHTML(text) {

    const div =
        document.createElement("div");

    div.textContent = text;

    return div.innerHTML;

}


/* =========================================================
   19. TOPBAR THEME BUTTON
   ========================================================= */

const themeButton =
    document.querySelector(".theme-btn");


if (themeButton) {

    themeButton.addEventListener(
        "click",
        function () {

            document.body.classList.toggle(
                "light-mode"
            );

        }
    );
}


/* =========================================================
   20. PREVENT EMPTY DEMO LOGIN
   ========================================================= */

console.log(
    "TruthLens loaded successfully 🚀"
);