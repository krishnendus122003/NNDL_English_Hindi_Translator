/*
============================================================
TRANSORA
English to Hindi Neural Machine Translation
Frontend Controller
============================================================
*/


// ==========================================================
// API
// ==========================================================

const API_BASE = "http://127.0.0.1:8000";



// ==========================================================
// DOM ELEMENTS
// ==========================================================

const connection = document.getElementById("connection");

const modelSelect = document.getElementById("model");
const decodingSelect = document.getElementById("decoding");
const decodingControl = document.getElementById("decoding-control");

const textInput = document.getElementById("text");
const count = document.getElementById("count");

const output = document.getElementById("output");
const metadata = document.getElementById("metadata");

const clearButton = document.getElementById("clear");
const copyButton = document.getElementById("copy");

const warningBox = document.getElementById("warning");
const errorBox = document.getElementById("error");

const translateButton = document.getElementById("translate");
const compareButton = document.getElementById("compare");
const attentionButton = document.getElementById("attention");

const includeNllb = document.getElementById("include-nllb");

const loading = document.getElementById("loading");

const comparisonPanel = document.getElementById("comparison");
const compareResults = document.getElementById("compare-results");

const alignmentPanel = document.getElementById("alignment");

const attentionTranslation =
    document.getElementById("attention-translation");

const heatmap =
    document.getElementById("heatmap");

const exampleButtons =
    document.querySelectorAll("[data-example]");



// ==========================================================
// APP STATE
// ==========================================================

let lastTranslation = "";

let apiConnected = false;

let nllbAvailable = false;



// ==========================================================
// HELPERS
// ==========================================================

function clearMessages() {

    warningBox.hidden = true;
    warningBox.textContent = "";

    errorBox.hidden = true;
    errorBox.textContent = "";
}



function showWarning(message) {

    if (!message) {

        warningBox.hidden = true;
        warningBox.textContent = "";

        return;
    }

    warningBox.textContent = message;
    warningBox.hidden = false;
}



function showError(message) {

    errorBox.textContent = message;
    errorBox.hidden = false;
}



function setLoading(
    active,
    message = "Translating..."
) {

    loading.hidden = !active;

    loading.textContent = message;

    translateButton.disabled = active;
    compareButton.disabled = active;
    attentionButton.disabled = active;

    modelSelect.disabled = active;
    decodingSelect.disabled = active;
}



function updateCharacterCount() {

    count.textContent =
        `${textInput.value.length.toLocaleString()} / 20,000`;
}



function getInputText() {

    return textInput.value.trim();
}



// ==========================================================
// CLEAR TRANSLATION
// ==========================================================

function clearTranslation() {

    lastTranslation = "";

    output.textContent =
        "Your translation will appear here.";

    output.classList.add("placeholder");

    metadata.textContent = "Ready";

    copyButton.disabled = true;

    clearMessages();
}



// ==========================================================
// MODEL UI
// ==========================================================

function updateModelUI() {

    const selectedModel =
        modelSelect.value;


    /*
    GRU and GRU + Attention use greedy decoding.
    */

    if (
        selectedModel === "gru" ||
        selectedModel === "attention"
    ) {

        decodingSelect.value = "greedy";

        decodingControl.hidden = true;

    } else {

        decodingControl.hidden = false;
    }


    /*
    Enable / disable NLLB depending on backend availability.
    */

    const nllbOption =
        modelSelect.querySelector(
            'option[value="nllb"]'
        );


    if (nllbOption) {

        nllbOption.disabled =
            !nllbAvailable;
    }
}



// ==========================================================
// API ERROR
// ==========================================================

async function readApiError(response) {

    try {

        const body =
            await response.json();


        if (body.detail) {

            if (
                typeof body.detail === "string"
            ) {

                return body.detail;
            }

            return JSON.stringify(
                body.detail
            );
        }


        return `Request failed (${response.status}).`;


    } catch {

        return `Request failed (${response.status}).`;
    }
}



// ==========================================================
// CHECK BACKEND
// ==========================================================

async function checkBackend() {

    try {

        const response =
            await fetch(
                `${API_BASE}/health`
            );


        if (!response.ok) {

            throw new Error(
                `HTTP ${response.status}`
            );
        }


        const health =
            await response.json();


        apiConnected = true;


        /*
        Keep navbar clean.
        */

        connection.textContent =
            "Connected";


        /*
        Find NLLB availability.
        */

        const models =
            health.models || [];


        const nllb =
            models.find(
                item =>
                    item.name === "nllb"
            );


        nllbAvailable =
            Boolean(
                nllb &&
                nllb.available
            );


        const nllbOption =
            modelSelect.querySelector(
                'option[value="nllb"]'
            );


        if (nllbOption) {

            nllbOption.disabled =
                !nllbAvailable;
        }


        includeNllb.disabled =
            !nllbAvailable;


        if (!nllbAvailable) {

            includeNllb.checked = false;


            if (
                modelSelect.value ===
                "nllb"
            ) {

                modelSelect.value =
                    "transformer";
            }
        }


        updateModelUI();


    } catch (error) {

        apiConnected = false;

        connection.textContent =
            "Offline";


        showError(
            "Cannot connect to the translation server."
        );
    }
}



// ==========================================================
// TRANSLATE
// ==========================================================

async function translateText() {

    clearMessages();


    const text =
        getInputText();


    if (!text) {

        showError(
            "Please enter an English sentence."
        );

        textInput.focus();

        return;
    }


    if (!apiConnected) {

        showError(
            "Translation server is not connected."
        );

        return;
    }


    const selectedModel =
        modelSelect.value;


    /*
    Never fall back to another model
    when NLLB is selected.
    */

    if (
        selectedModel === "nllb" &&
        !nllbAvailable
    ) {

        clearTranslation();

        showError(
            "NLLB is currently unavailable."
        );

        return;
    }


    let decoding =
        decodingSelect.value;


    if (
        selectedModel === "gru" ||
        selectedModel === "attention"
    ) {

        decoding = "greedy";
    }


    setLoading(
        true,
        "Translating..."
    );


    try {

        const response =
            await fetch(
                `${API_BASE}/translate`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            text: text,
                            model:
                                selectedModel,
                            decoding:
                                decoding
                        })
                }
            );


        if (!response.ok) {

            const message =
                await readApiError(
                    response
                );


            clearTranslation();

            throw new Error(message);
        }


        const result =
            await response.json();


        lastTranslation =
            result.translation || "";


        output.textContent =
            lastTranslation ||
            "No translation produced.";


        output.classList.remove(
            "placeholder"
        );


        /*
        Keep result metadata small and useful.
        */

        metadata.textContent =
            `${result.decoding} · ` +
            `${Number(
                result.latency_ms
            ).toLocaleString()} ms`;


        copyButton.disabled =
            !lastTranslation;


        showWarning(
            result.warning
        );


    } catch (error) {

        showError(
            error.message ||
            "Translation failed."
        );


    } finally {

        setLoading(false);

        updateModelUI();
    }
}



// ==========================================================
// COMPARE MODELS
// ==========================================================

async function compareModels() {

    clearMessages();


    const text =
        getInputText();


    if (!text) {

        showError(
            "Please enter an English sentence first."
        );

        textInput.focus();

        return;
    }


    if (!apiConnected) {

        showError(
            "Translation server is not connected."
        );

        return;
    }


    setLoading(
        true,
        "Comparing..."
    );


    compareResults.innerHTML = "";


    try {

        const response =
            await fetch(
                `${API_BASE}/compare`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({

                            text: text,

                            decoding:
                                decodingSelect.value,

                            include_nllb:
                                includeNllb.checked
                        })
                }
            );


        if (!response.ok) {

            throw new Error(
                await readApiError(
                    response
                )
            );
        }


        const data =
            await response.json();


        comparisonPanel.hidden =
            false;


        renderComparison(
            data.results || []
        );


        comparisonPanel.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });


    } catch (error) {

        showError(
            error.message ||
            "Model comparison failed."
        );


    } finally {

        setLoading(false);

        updateModelUI();
    }
}



// ==========================================================
// RENDER COMPARISON
// ==========================================================

function renderComparison(results) {

    compareResults.innerHTML = "";


    if (!results.length) {

        compareResults.textContent =
            "No results returned.";

        return;
    }


    results.forEach(result => {

        const card =
            document.createElement(
                "article"
            );


        card.className =
            "compare-card";


        const heading =
            document.createElement(
                "h3"
            );


        /*
        Use short clean model names.
        */

        const names = {

            gru:
                "GRU",

            attention:
                "GRU + Attention",

            transformer:
                "Transformer",

            nllb:
                "NLLB"
        };


        heading.textContent =
            names[result.model] ||
            result.model ||
            "Model";


        card.appendChild(
            heading
        );


        if (result.success) {

            const translation =
                document.createElement(
                    "p"
                );


            translation.lang = "hi";


            translation.textContent =
                result.translation ||
                "No translation produced.";


            card.appendChild(
                translation
            );


            const meta =
                document.createElement(
                    "small"
                );


            meta.textContent =
                `${result.decoding} · ` +
                `${Number(
                    result.latency_ms
                ).toLocaleString()} ms`;


            card.appendChild(meta);


            if (result.warning) {

                const warning =
                    document.createElement(
                        "p"
                    );


                warning.className =
                    "small";


                warning.textContent =
                    result.warning;


                card.appendChild(
                    warning
                );
            }


        } else {

            const failure =
                document.createElement(
                    "p"
                );


            failure.textContent =
                result.warning ||
                "Model unavailable.";


            failure.className =
                "error-text";


            card.appendChild(
                failure
            );
        }


        compareResults.appendChild(
            card
        );
    });
}



// ==========================================================
// ATTENTION
// ==========================================================

async function showAttention() {

    clearMessages();


    const text =
        getInputText();


    if (!text) {

        showError(
            "Please enter an English sentence first."
        );

        textInput.focus();

        return;
    }


    if (!apiConnected) {

        showError(
            "Translation server is not connected."
        );

        return;
    }


    setLoading(
        true,
        "Generating attention..."
    );


    try {

        const response =
            await fetch(
                `${API_BASE}/attention`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            text: text
                        })
                }
            );


        if (!response.ok) {

            throw new Error(
                await readApiError(
                    response
                )
            );
        }


        const result =
            await response.json();


        attentionTranslation.textContent =
            result.translation || "";


        renderHeatmap(

            result.source_tokens || [],

            result.target_tokens || [],

            result.attention_matrix || []

        );


        alignmentPanel.hidden =
            false;


        alignmentPanel.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });


        showWarning(
            result.warning
        );


    } catch (error) {

        showError(
            error.message ||
            "Could not generate attention."
        );


    } finally {

        setLoading(false);

        updateModelUI();
    }
}



// ==========================================================
// ATTENTION HEATMAP
// ==========================================================

function renderHeatmap(
    sourceTokens,
    targetTokens,
    matrix
) {

    heatmap.innerHTML = "";


    if (
        !sourceTokens.length ||
        !targetTokens.length ||
        !matrix.length
    ) {

        heatmap.textContent =
            "No attention data returned.";

        return;
    }


    /*
    Header
    */

    const headerRow =
        document.createElement("tr");


    const emptyCorner =
        document.createElement("th");


    headerRow.appendChild(
        emptyCorner
    );


    sourceTokens.forEach(token => {

        const th =
            document.createElement(
                "th"
            );


        th.scope = "col";

        th.textContent =
            readablePiece(token);


        headerRow.appendChild(th);
    });


    heatmap.appendChild(
        headerRow
    );


    /*
    Rows
    */

    const rowCount =
        Math.min(
            targetTokens.length,
            matrix.length
        );


    for (
        let rowIndex = 0;
        rowIndex < rowCount;
        rowIndex++
    ) {

        const row =
            document.createElement(
                "tr"
            );


        const label =
            document.createElement(
                "th"
            );


        label.scope = "row";


        label.textContent =
            readablePiece(
                targetTokens[
                    rowIndex
                ]
            );


        row.appendChild(label);


        const weights =
            matrix[rowIndex] || [];


        sourceTokens.forEach(
            (_, columnIndex) => {

                const cell =
                    document.createElement(
                        "td"
                    );


                const rawWeight =
                    Number(
                        weights[
                            columnIndex
                        ] || 0
                    );


                const weight =
                    Math.max(
                        0,
                        Math.min(
                            1,
                            rawWeight
                        )
                    );


                cell.textContent =
                    weight.toFixed(2);


                cell.style.backgroundColor =
                    `rgba(
                        0,
                        105,
                        95,
                        ${0.08 + weight * 0.82}
                    )`;


                cell.title =
                    `Attention: ${weight.toFixed(4)}`;


                row.appendChild(
                    cell
                );
            }
        );


        heatmap.appendChild(row);
    }
}



// ==========================================================
// SENTENCEPIECE DISPLAY
// ==========================================================

function readablePiece(piece) {

    if (piece === "<s>") {

        return "BOS";
    }


    if (piece === "</s>") {

        return "EOS";
    }


    if (piece === "<pad>") {

        return "PAD";
    }


    return piece.replace(
        /▁/g,
        "·"
    );
}



// ==========================================================
// COPY
// ==========================================================

async function copyTranslation() {

    if (!lastTranslation) {

        return;
    }


    try {

        await navigator.clipboard.writeText(
            lastTranslation
        );


        const oldText =
            copyButton.textContent;


        copyButton.textContent =
            "Copied";


        setTimeout(
            () => {

                copyButton.textContent =
                    oldText;

            },
            1200
        );


    } catch {

        showError(
            "Could not copy translation."
        );
    }
}



// ==========================================================
// CLEAR EVERYTHING
// ==========================================================

function clearEverything() {

    textInput.value = "";


    updateCharacterCount();


    clearTranslation();


    comparisonPanel.hidden =
        true;


    alignmentPanel.hidden =
        true;


    compareResults.innerHTML =
        "";


    heatmap.innerHTML =
        "";


    textInput.focus();
}



// ==========================================================
// EXAMPLES
// ==========================================================

exampleButtons.forEach(
    button => {

        button.addEventListener(
            "click",
            () => {

                textInput.value =
                    button.dataset
                        .example || "";


                updateCharacterCount();


                clearTranslation();


                comparisonPanel.hidden =
                    true;


                alignmentPanel.hidden =
                    true;


                textInput.focus();
            }
        );
    }
);



// ==========================================================
// EVENTS
// ==========================================================

textInput.addEventListener(
    "input",
    updateCharacterCount
);



translateButton.addEventListener(
    "click",
    translateText
);



compareButton.addEventListener(
    "click",
    compareModels
);



attentionButton.addEventListener(
    "click",
    showAttention
);



copyButton.addEventListener(
    "click",
    copyTranslation
);



clearButton.addEventListener(
    "click",
    clearEverything
);



modelSelect.addEventListener(
    "change",
    () => {

        /*
        Clear old output when model changes.
        */

        clearTranslation();

        comparisonPanel.hidden =
            true;

        alignmentPanel.hidden =
            true;

        updateModelUI();
    }
);



decodingSelect.addEventListener(
    "change",
    () => {

        /*
        Clear old result when decoding changes.
        */

        clearTranslation();
    }
);



// ==========================================================
// CTRL + ENTER
// ==========================================================

textInput.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter" &&
            (
                event.ctrlKey ||
                event.metaKey
            )
        ) {

            event.preventDefault();

            translateText();
        }
    }
);



// ==========================================================
// INITIALISE
// ==========================================================

async function initialise() {

    updateCharacterCount();

    clearTranslation();

    updateModelUI();

    await checkBackend();
}



initialise();