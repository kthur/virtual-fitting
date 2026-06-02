const axios = require('axios');
const fs = require('fs');
const path = require('path');
const { client, handle_file } = require('@gradio/client');

async function testRealVTON() {
  // 1. Download a real sample person image from IDM-VTON examples
  console.log('[1/4] Downloading sample person image...');
  const personUrl = 'https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/example/human/00034_00.jpg';
  const personRes = await axios.get(personUrl, { responseType: 'arraybuffer' });
  const personPath = path.join(__dirname, 'test_person.jpg');
  fs.writeFileSync(personPath, personRes.data);
  console.log('  -> Saved (' + personRes.data.length + ' bytes)');

  // 2. Clothing image from Musinsa
  const clothingUrl = 'https://image.msscdn.net/images/goods_img/20260406/6263413/6263413_17788097542676_500.jpg';
  console.log('[2/4] Clothing URL:', clothingUrl);

  // 3. Connect to Gradio
  console.log('[3/4] Connecting to IDM-VTON...');
  const hfApp = await client("yisol/IDM-VTON");
  console.log('  -> Connected!');

  // 4. Run prediction
  console.log('[4/4] Running AI prediction (20-60 seconds)...');
  const startTime = Date.now();
  
  const result = await hfApp.predict("/tryon", [
    { background: handle_file(personPath), layers: [], composite: null },
    handle_file(clothingUrl),
    "a garment",
    true,   // auto-masking
    false,  // auto-crop
    30,     // denoising steps
    42,     // seed
  ]);

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`  -> Done in ${elapsed}s!`);

  if (result && result.data && result.data[0]) {
    console.log('SUCCESS! Result:', JSON.stringify(result.data[0]).substring(0, 200));
  } else {
    console.log('UNEXPECTED:', JSON.stringify(result).substring(0, 300));
  }

  // Cleanup
  fs.unlinkSync(personPath);
}

testRealVTON().catch(e => {
  console.error('FAILED:', e.message || JSON.stringify(e));
});
