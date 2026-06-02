const express = require('express');
const cors = require('cors');
const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

// Path to the venv python with AI dependencies installed
const VENV_PYTHON = path.join(__dirname, '..', 'venv', 'Scripts', 'python.exe');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Basic health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Scraper endpoint
app.post('/api/scrape', async (req, res) => {
  const { url } = req.body;
  if (!url) {
    return res.status(400).json({ error: 'URL is required' });
  }

  try {
    // Fast-path: if URL is already an image
    if (url.match(/\.(jpeg|jpg|gif|png|webp)$/i)) {
      return res.json({ imageUrls: [url], garmentDescription: 'a garment', sourceUrl: url });
    }

    // 1. Fetch the HTML with a 10-second timeout
    const response = await axios.get(url, {
      timeout: 10000,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9,ko;q=0.8',
      }
    });
    
    const html = response.data;
    const $ = cheerio.load(html);

    // 2. Extract multiple images and text
    const images = new Set();
    
    // Open Graph image
    const ogImg = $('meta[property="og:image"]').attr('content');
    if (ogImg) images.add(ogImg);
    
    // Twitter card image
    const twImg = $('meta[name="twitter:image"]').attr('content');
    if (twImg) images.add(twImg);
    
    // Main images in document
    $('img').each((i, el) => {
      let src = $(el).attr('src');
      if (src && !src.includes('icon') && !src.includes('logo') && src.length > 20) {
        images.add(src);
      }
    });

    let imageUrls = Array.from(images).slice(0, 10);
    // Ensure absolute URLs
    imageUrls = imageUrls.map(img => {
      if (img.startsWith('//')) return 'https:' + img;
      if (!img.startsWith('http')) {
        const urlObj = new URL(url);
        return urlObj.protocol + '//' + urlObj.host + (img.startsWith('/') ? '' : '/') + img;
      }
      return img;
    });

    if (imageUrls.length === 0) {
      return res.status(404).json({ error: 'Could not find a product image on the provided URL.' });
    }

    // 3. Extract description (Title + Meta description)
    const title = $('title').text().trim();
    const metaDesc = $('meta[name="description"]').attr('content') || '';
    const garmentDescription = `${title} ${metaDesc}`.substring(0, 200).trim();

    res.json({ imageUrls, garmentDescription, sourceUrl: url });

  } catch (error) {
    console.error('Scraping error:', error.message);
    res.status(500).json({ error: 'Failed to scrape the URL. It might be blocking requests.' });
  }
});

// High-Quality Local AI Try-On
app.post('/api/tryon', async (req, res) => {
  const { userImageBase64, clothingImageUrl, garmentDescription } = req.body;
  if (!userImageBase64 || !clothingImageUrl) {
    return res.status(400).json({ error: 'Missing required images.' });
  }

  const tempUserPath = path.join(__dirname, `temp_user_${Date.now()}.png`);
  const tempGarmentPath = path.join(__dirname, `temp_garment_${Date.now()}.png`);
  const tempOutputPath = path.join(__dirname, `temp_output_${Date.now()}.png`);

  try {
    // 1. Decode base64 user image and save to temp file
    const base64Data = userImageBase64.replace(/^data:image\/\w+;base64,/, "");
    fs.writeFileSync(tempUserPath, base64Data, 'base64');

    // 2. Download clothing image and save to temp file
    console.log(`[Server] Downloading garment image from: ${clothingImageUrl}`);
    const response = await axios({
      url: clothingImageUrl,
      method: 'GET',
      responseType: 'arraybuffer',
      timeout: 20000,
    });
    fs.writeFileSync(tempGarmentPath, Buffer.from(response.data));

    // 3. Spawn Python process to run local try-on
    const tryonScriptPath = path.join(__dirname, 'tryon_local.py');
    console.log(`[Server] Spawning local AI process: ${VENV_PYTHON} ${tryonScriptPath}`);
    const { spawn } = require('child_process');
    const pythonProcess = spawn(VENV_PYTHON, [
      tryonScriptPath,
      '--person_path', tempUserPath,
      '--garment_path', tempGarmentPath,
      '--output_path', tempOutputPath,
      '--prompt', garmentDescription || 'a garment'
    ], { cwd: __dirname });

    let stdoutLogs = '';
    let stderrLogs = '';

    pythonProcess.stdout.on('data', (data) => {
      const chunk = data.toString();
      stdoutLogs += chunk;
      console.log(chunk.trim());
    });

    pythonProcess.stderr.on('data', (data) => {
      const chunk = data.toString();
      stderrLogs += chunk;
      console.error(chunk.trim());
    });

    const exitCode = await new Promise((resolve) => {
      pythonProcess.on('close', resolve);
    });

    console.log(`[Server] AI process closed with exit code: ${exitCode}`);

    if (exitCode !== 0) {
      throw new Error(`Local AI Try-On process failed with exit code ${exitCode}.\nStderr: ${stderrLogs}`);
    }

    // 4. Read result image, encode to base64 and return
    if (!fs.existsSync(tempOutputPath)) {
      throw new Error(`Output image was not generated by the AI script.`);
    }

    const outputBuffer = fs.readFileSync(tempOutputPath);
    const base64Output = outputBuffer.toString('base64');
    const resultImageUrl = `data:image/png;base64,${base64Output}`;

    res.json({ resultImageUrl });

  } catch (error) {
    console.error('[Server] VTON error:', error);
    res.status(500).json({ error: 'Failed to process local AI Try-On. ' + error.message });
  } finally {
    // Cleanup all temp files
    [tempUserPath, tempGarmentPath, tempOutputPath].forEach(filePath => {
      try {
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      } catch (err) {
        console.error(`[Server] Temp cleanup failed for ${filePath}:`, err.message);
      }
    });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Backend server running on http://0.0.0.0:${PORT}`);
});
