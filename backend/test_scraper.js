const axios = require('axios');
const cheerio = require('cheerio');

async function testScrape() {
  const url = 'https://www.musinsa.com/products/6263413';
  try {
    const response = await axios.get(url, {
      timeout: 10000,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'en-US,en;q=0.9,ko;q=0.8',
      }
    });
    
    const $ = cheerio.load(response.data);
    
    // Images
    const images = [];
    $('img').each((i, el) => {
      const src = $(el).attr('src');
      if (src && src.startsWith('http') && !src.includes('icon')) {
        images.push(src);
      }
    });
    
    // Description / body info
    const title = $('title').text();
    const metaDesc = $('meta[name="description"]').attr('content');
    
    console.log('Title:', title);
    console.log('Meta Desc:', metaDesc);
    console.log('First 5 Images:', images.slice(0, 5));
    
  } catch (e) {
    console.log('Error:', e.message);
  }
}
testScrape();
