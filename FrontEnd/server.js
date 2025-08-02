const express = require("express");
const path = require("path");
const axios = require("axios");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 3000;
const FLASK_BASE_URL = process.env.FLASK_BASE_URL || "http://localhost:5000";

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, "public")));

// Handle file uploads
app.use(express.raw({ type: 'multipart/form-data', limit: '50mb' }));

app.get("/", (req, res) => {
    console.log(`[${new Date().toISOString()}] GET / - Serving UI`);
    res.sendFile(path.join(__dirname, "views", "ui.html"));
});

app.post("/search", async (req, res) => {
    const { query, limit, requirePdf, summarize } = req.body;
    const requestTimestamp = new Date().toISOString();
    console.log(
        `[${requestTimestamp}] POST /search - Query: "${query}", limit: ${limit || "default"}, requirePdf: ${requirePdf}, summarize: ${summarize}`
    );

    if (!query) {
        console.log(`[${requestTimestamp}] POST /search - Error: Query missing`);
        return res.status(400).json({ error: "Query parameter is required." });
    }

    const validatedLimit = parseInt(limit, 10) || 15;
    if (validatedLimit <= 0 || validatedLimit > 100) {
        console.log(`[${requestTimestamp}] POST /search - Error: Invalid limit ${limit}`);
        return res.status(400).json({ error: "Limit must be between 1 and 100." });
    }

    const maxRetries = 3;
    let attempt = 0;
    while (attempt < maxRetries) {
        try {
            let targetUrl = `${FLASK_BASE_URL}/search`;
            if (requirePdf === true) {
                targetUrl += "?require_pdf=true";
            }

            console.log(`[${requestTimestamp}] POST /search - Attempt ${attempt + 1}: ${targetUrl}`);

            const flaskResponse = await axios.post(
                targetUrl,
                { query, limit: validatedLimit, summarize },
                { timeout: 60000 }
            );

            console.log(`[${requestTimestamp}] POST /search - Success (Status: ${flaskResponse.status})`);
            return res.status(flaskResponse.status).json(flaskResponse.data);
        } catch (error) {
            attempt++;
            console.error(`[${requestTimestamp}] POST /search - Attempt ${attempt} failed: ${error.message}`);
            if (attempt === maxRetries) {
                if (error.response) {
                    return res.status(error.response.status).json({
                        error: error.response.data?.error || `Flask API error (${error.response.status})`
                    });
                } else if (error.code === "ECONNABORTED") {
                    return res.status(504).json({ error: "The request to the search service timed out." });
                } else {
                    return res.status(502).json({ error: "Could not connect to the search service." });
                }
            }
            await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, attempt)));
        }
    }
});

app.post("/upload", async (req, res) => {
    const requestTimestamp = new Date().toISOString();
    console.log(`[${requestTimestamp}] POST /upload - Received file upload`);

    const maxRetries = 3;
    let attempt = 0;
    while (attempt < maxRetries) {
        try {
            const response = await axios.post(
                `${FLASK_BASE_URL}/upload`,
                req.body,
                {
                    headers: { 'Content-Type': req.headers['content-type'] },
                    timeout: 120000
                }
            );

            console.log(`[${requestTimestamp}] POST /upload - Success (Status: ${response.status})`);
            return res.status(response.status).json(response.data);
        } catch (error) {
            attempt++;
            console.error(`[${requestTimestamp}] POST /upload - Attempt ${attempt} failed: ${error.message}`);
            if (attempt === maxRetries) {
                if (error.response) {
                    return res.status(error.response.status).json({
                        error: error.response.data?.error || `Flask API error (${error.response.status})`
                    });
                } else if (error.code === "ECONNABORTED") {
                    return res.status(504).json({ error: "The upload request timed out." });
                } else {
                    return res.status(502).json({ error: "Could not connect to the upload service." });
                }
            }
            await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, attempt)));
        }
    }
});

app.listen(PORT, async () => {
    console.log(`[${new Date().toISOString()}] Server started on port ${PORT}`);
    console.log(`Flask API: ${FLASK_BASE_URL}`);
    console.log(`Frontend: http://localhost:${PORT}`);
    try {
        await axios.get(`${FLASK_BASE_URL}/test_search`, { timeout: 5000 });
        console.log(`[${new Date().toISOString()}] Flask API is accessible`);
    } catch (error) {
        console.error(`[${new Date().toISOString()}] Flask API test failed: ${error.message}`);
    }
});