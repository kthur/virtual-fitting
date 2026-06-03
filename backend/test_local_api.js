const axios = require('axios');
const fs = require('fs');
const path = require('path');

async function testLocalApi() {
  console.log('[1/4] Preparing test images...');
  
  // Load person image and encode to base64
  const personPath = path.join(__dirname, 'test_person.jpg');
  if (!fs.existsSync(personPath)) {
    throw new Error(`test_person.jpg not found at ${personPath}. Please run the python test first.`);
  }
  const personBase64 = fs.readFileSync(personPath).toString('base64');
  const userImageBase64 = `data:image/jpeg;base64,${personBase64}`;
  
  // Use a real garment image URL
  const clothingImageUrl = 'https://image.msscdn.net/images/goods_img/20260406/6263413/6263413_17788097542676_500.jpg';
  const garmentDescription = 'a grey hoodie sweater';

  console.log('[2/4] Sending request to local backend API at http://localhost:3000/api/tryon...');
  const startTime = Date.now();
  
  try {
    const response = await axios.post('http://localhost:3000/api/tryon', {
      userImageBase64,
      clothingImageUrl,
      garmentDescription,
      garmentType: 'upper',
      fitType: 'overfit'
    }, {
      timeout: 180000 // 3 minutes timeout for CPU/GPU inference
    });

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[3/4] API response received in ${elapsed}s!`);

    if (response.data && response.data.resultImageUrl) {
      console.log('[4/4] Saving output image...');
      const base64Data = response.data.resultImageUrl.replace(/^data:image\/\w+;base64,/, "");
      const outputPath = path.join(__dirname, 'test_api_output.png');
      fs.writeFileSync(outputPath, base64Data, 'base64');
      console.log(`SUCCESS! Result saved to ${outputPath}`);
    } else {
      console.log('FAILED: No resultImageUrl in response', response.data);
    }
  } catch (error) {
    console.error('API Request failed:', error.response ? error.response.data : error.message);
  }
}

testLocalApi();
