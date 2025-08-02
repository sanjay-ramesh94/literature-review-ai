// server.js
const express = require("express");
const path = require("path");
const axios = require("axios");
const multer = require("multer"); // For handling file uploads
const FormData = require("form-data"); // For forwarding files to Flask
const fs = require("fs"); // File system module
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 3000; // Use PORT env var if available

// Configuration for Flask API connection
const FLASK_BASE_URL = process.env.FLASK_BASE_URL || "http://127.0.0.1:5000"; // Use 127.0.0.1
const FLASK_SEARCH_ENDPOINT = "/search";
const FLASK_PROCESS_FILE_ENDPOINT = "/process-and-generate"; // Endpoint for file processing
const FLASK_PROCESS_URL_ENDPOINT = "/process-url-and-generate"; // Endpoint for URL processing

// --- Middleware ---
app.use(express.json()); // For parsing application/json
app.use(express.urlencoded({ extended: true })); // For parsing application/x-www-form-urlencoded

// Multer configuration: Store file in memory buffer
const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB file size limit
});

// --- Static File Serving (Optional but Recommended) ---
// Serve static files (CSS, JS) if you separate them later
// app.use(express.static(path.join(__dirname, 'public')));

// --- Routes ---

// Function to get current timestamp for logging
const ts = () => `[${new Date().toISOString()}]`;

// Serve the main frontend UI
app.get("/", (req, res) => {
  console.log(`${ts()} GET / - Serving UI`);
  // Define view directory relative to this file's location
  const viewsDir = path.join(__dirname, "views");
  const uiPath = path.join(viewsDir, "ui.html");

  if (!fs.existsSync(uiPath)) {
    console.error(`${ts()} GET / - Error: ui.html not found at ${uiPath}`);
    return res.status(404).send("Main UI file not found.");
  }
  res.sendFile(uiPath);
});

// Serve the review page
app.get("/review", (req, res) => {
  console.log(`${ts()} GET /review - Serving review page`);
  const viewsDir = path.join(__dirname, "views");
  const reviewPath = path.join(viewsDir, "review.html");

  if (!fs.existsSync(reviewPath)) {
    console.error(
      `${ts()} GET /review - Error: review not found at ${reviewPath}`
    );
    return res.status(404).send("Review page file not found.");
  }
  res.sendFile(reviewPath);
});

// Handle paper search requests from frontend -> forward to Flask /search
app.post("/search", async (req, res) => {
  const { query, limit, requirePdf } = req.body;
  console.log(
    `${ts()} POST /search - Received query: "${query}", limit: ${
      limit || "default"
    }, requirePdf: ${requirePdf}`
  );

  if (!query) {
    console.log(`${ts()} POST /search - Error: Query missing`);
    return res.status(400).json({ error: "Query parameter is required." });
  }
  const validatedLimit = parseInt(limit, 10) || 10; // Default 10
  if (validatedLimit <= 0 || validatedLimit > 50) {
    // Max 50
    console.log(`${ts()} POST /search - Error: Invalid limit ${limit}`);
    return res.status(400).json({ error: "Limit must be between 1 and 50." });
  }

  try {
    let targetUrl = FLASK_BASE_URL + FLASK_SEARCH_ENDPOINT;
    if (requirePdf === true) {
      targetUrl += "?require_pdf=true"; // Append query parameter correctly
      console.log(`${ts()} POST /search - Appending ?require_pdf=true`);
    }
    console.log(`${ts()} POST /search - Forwarding to Flask: ${targetUrl}`);

    const flaskResponse = await axios.post(
      targetUrl,
      { query: query, limit: validatedLimit }, // Body sent to Flask
      { timeout: 45000 } // 45 second timeout for search
    );

    console.log(
      `${ts()} POST /search - Flask response status: ${flaskResponse.status}`
    );
    res.status(flaskResponse.status).json(flaskResponse.data);
  } catch (error) {
    handleFlaskError(error, res, ts(), "/search", "search service");
  }
});

// --- Review Generation Routes ---

// Handle review generation from FILE UPLOAD
app.post("/generate-review", upload.single("pdfFile"), async (req, res) => {
  console.log(`${ts()} POST /generate-review - Received request.`);

  if (!req.file) {
    console.log(`${ts()} POST /generate-review - Error: No file uploaded.`);
    return res.status(400).json({ error: "No PDF file uploaded." });
  }
  if (req.file.mimetype !== "application/pdf") {
    console.log(
      `${ts()} POST /generate-review - Error: Uploaded file is not a PDF (${
        req.file.mimetype
      }).`
    );
    return res
      .status(400)
      .json({
        error: `Invalid file type: ${req.file.mimetype}. Please upload a PDF.`,
      });
  }

  console.log(
    `${ts()} POST /generate-review - Received file: ${
      req.file.originalname
    }, Size: ${req.file.size} bytes`
  );

  try {
    const form = new FormData();
    form.append("pdfFile", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const flaskProcessUrl = FLASK_BASE_URL + FLASK_PROCESS_FILE_ENDPOINT;
    console.log(
      `${ts()} POST /generate-review - Forwarding file to Flask: ${flaskProcessUrl}`
    );

    const flaskResponse = await axios.post(flaskProcessUrl, form, {
      headers: { ...form.getHeaders() },
      timeout: 300000, // Increased timeout (5 minutes) for potentially long processing + LLM
    });

    console.log(
      `${ts()} POST /generate-review - Flask response status: ${
        flaskResponse.status
      }`
    );
    res.status(flaskResponse.status).json(flaskResponse.data);
  } catch (error) {
    handleFlaskError(
      error,
      res,
      ts(),
      "/generate-review",
      "review generation service"
    );
  }
});

// Handle review generation from PDF URL
app.post("/generate-review-url", async (req, res) => {
  console.log(`${ts()} POST /generate-review-url - Received request.`);
  const { pdfUrl } = req.body;

  if (!pdfUrl || typeof pdfUrl !== "string") {
    console.log(
      `${ts()} POST /generate-review-url - Error: Missing or invalid pdfUrl.`
    );
    return res
      .status(400)
      .json({ error: "Missing or invalid pdfUrl in request body." });
  }
  // Simple URL validation
  try {
    new URL(pdfUrl);
  } catch (_) {
    console.log(
      `${ts()} POST /generate-review-url - Error: Invalid URL format: ${pdfUrl}`
    );
    return res.status(400).json({ error: "Invalid URL format provided." });
  }

  console.log(`${ts()} POST /generate-review-url - Received URL: ${pdfUrl}`);

  try {
    const flaskProcessUrl = FLASK_BASE_URL + FLASK_PROCESS_URL_ENDPOINT;
    console.log(
      `${ts()} POST /generate-review-url - Forwarding URL to Flask: ${flaskProcessUrl}`
    );

    const flaskResponse = await axios.post(
      flaskProcessUrl,
      { pdfUrl: pdfUrl }, // Send as JSON
      {
        headers: { "Content-Type": "application/json" },
        timeout: 300000, // Long timeout (5 minutes)
      }
    );

    console.log(
      `${ts()} POST /generate-review-url - Flask response status: ${
        flaskResponse.status
      }`
    );
    res.status(flaskResponse.status).json(flaskResponse.data);
  } catch (error) {
    handleFlaskError(
      error,
      res,
      ts(),
      "/generate-review-url",
      "review generation service"
    );
  }
});

// --- Reusable Error Handler for Axios calls to Flask ---
function handleFlaskError(
  error,
  res,
  timestamp,
  routeName,
  serviceName = "backend service"
) {
  console.error(
    `${timestamp} POST ${routeName} - Error during Flask call:`,
    error.message
  );
  if (error.response) {
    console.error(
      `${timestamp} POST ${routeName} - Flask Error Status:`,
      error.response.status
    );
    console.error(
      `${timestamp} POST ${routeName} - Flask Error Data:`,
      JSON.stringify(error.response.data)
    );
    // Try to return Flask's error message, otherwise provide a generic one
    res
      .status(error.response.status)
      .json({
        error:
          error.response.data?.error ||
          `Error from ${serviceName} (${error.response.status})`,
      });
  } else if (error.request) {
    console.error(
      `${timestamp} POST ${routeName} - No response received from Flask.`
    );
    res.status(502).json({ error: `Could not connect to the ${serviceName}.` }); // Bad Gateway
  } else if (error.code === "ECONNABORTED") {
    console.error(
      `${timestamp} POST ${routeName} - Request to Flask timed out.`
    );
    res
      .status(504)
      .json({ error: `The request to the ${serviceName} timed out.` }); // Gateway Timeout
  } else {
    console.error(
      `${timestamp} POST ${routeName} - Axios setup or other error:`,
      error.message
    );
    res
      .status(500)
      .json({
        error: `An internal server error occurred while contacting the ${serviceName}.`,
      });
  }
}

// --- Start Server ---
app.listen(PORT, () => {
  console.log(`${ts()} Express server started successfully.`);
  console.log(`${ts()} Flask API Base URL configured as: ${FLASK_BASE_URL}`);
  console.log(
    `${ts()} Frontend accessible at http://localhost:${PORT} or http://127.0.0.1:${PORT}`
  );
});
