const axios = require('axios');
const cheerio = require('cheerio');

/**
 * 쇼핑몰 URL을 감지하고 해당 쇼핑몰의 사이즈 표 스크래퍼로 라우팅합니다.
 * 현재 지원: 무신사, 지그재그, 29cm
 */

const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

function detectMall(url) {
  const u = url.toLowerCase();
  if (u.includes('musinsa.com')) return 'musinsa';
  if (u.includes('zigzag.kr'))  return 'zigzag';
  if (u.includes('29cm.co.kr')) return '29cm';
  return null;
}

// ─────────────────────────────────────────────────────────────
// 무신사 사이즈 표 스크래퍼
// 무신사는 __NEXT_DATA__ JSON에 sizeTable을 넣어두는 경우가 많음
// ─────────────────────────────────────────────────────────────
async function scrapeMusinsaSize(url) {
  const response = await axios.get(url, {
    timeout: 15000,
    headers: { 'User-Agent': USER_AGENT, 'Accept-Language': 'ko-KR,ko;q=0.9' }
  });
  const $ = cheerio.load(response.data);

  // 1) __NEXT_DATA__ JSON 우선 시도
  const nextDataText = $('#__NEXT_DATA__').html();
  if (nextDataText) {
    try {
      const nextData = JSON.parse(nextDataText);
      const meta = nextData?.props?.pageProps?.meta?.data;
      if (meta) {
        // 무신사 사이즈 표는 sizeTableByImg 아래에 있을 수 있음
        const candidates = [
          meta.sizeTable, meta.sizeTableByImg, meta.goodsSizeTable,
          meta.sizeInfo, meta.sizeChart
        ];
        for (const c of candidates) {
          if (Array.isArray(c) && c.length) {
            return normalizeSizeChart('musinsa', c);
          }
        }
      }
    } catch (e) { /* fall through */ }
  }

  // 2) HTML fallback: 사이즈 표 <table> 직접 파싱
  const sizeTables = [];
  $('table').each((_, tbl) => {
    const text = $(tbl).text();
    if (/사이즈|size|어깨|가슴|허리|총장/i.test(text)) {
      sizeTables.push(parseHtmlTable($, tbl));
    }
  });

  if (sizeTables.length) {
    // 가장 사이즈 항목이 많은 표 선택
    sizeTables.sort((a, b) => b.rows.length - a.rows.length);
    return normalizeSizeChart('musinsa', sizeTables[0]);
  }

  throw new Error('무신사 페이지에서 사이즈 표를 찾을 수 없습니다.');
}

// ─────────────────────────────────────────────────────────────
// 지그재그 사이즈 표 스크래퍼
// ─────────────────────────────────────────────────────────────
async function scrapeZigzagSize(url) {
  const response = await axios.get(url, {
    timeout: 15000,
    headers: { 'User-Agent': USER_AGENT, 'Accept-Language': 'ko-KR,ko;q=0.9' }
  });
  const $ = cheerio.load(response.data);

  // 지그재그는 SSR된 HTML에 사이즈 표가 <table> 또는 <div class*=size>로 있음
  const sizeBlocks = [];
  $('[class*="size" i]').each((_, el) => {
    const text = $(el).text();
    if (/어깨|가슴|소매|총장|허리|힙/i.test(text) && text.length < 4000) {
      sizeBlocks.push(text);
    }
  });

  if (sizeBlocks.length) {
    return { mall: 'zigzag', rawText: sizeBlocks[0] };
  }

  // fallback: 모든 table 파싱
  const tables = $('table').map((_, t) => parseHtmlTable($, t)).get();
  if (tables.length) {
    return normalizeSizeChart('zigzag', tables[0]);
  }

  throw new Error('지그재그 페이지에서 사이즈 표를 찾을 수 없습니다.');
}

// ─────────────────────────────────────────────────────────────
// 29cm 사이즈 표 스크래퍼
// ─────────────────────────────────────────────────────────────
async function scrape29cmSize(url) {
  const response = await axios.get(url, {
    timeout: 15000,
    headers: { 'User-Agent': USER_AGENT, 'Accept-Language': 'ko-KR,ko;q=0.9' }
  });
  const $ = cheerio.load(response.data);

  // 29cm은 PDP에 사이즈 표가 __INITIAL_STATE__ 또는 inline JSON으로 들어있음
  const scripts = $('script').map((_, s) => $(s).html() || '').get();
  for (const s of scripts) {
    const m = s.match(/size[A-Za-z]*\s*[:=]\s*(\[[\s\S]*?\])/);
    if (m) {
      try {
        const arr = JSON.parse(m[1]);
        if (Array.isArray(arr) && arr.length) {
          return normalizeSizeChart('29cm', arr);
        }
      } catch (e) { /* ignore */ }
    }
  }

  // HTML table fallback
  const tables = $('table').map((_, t) => parseHtmlTable($, t)).get()
    .filter(t => /어깨|가슴|소매|총장|허리|힙/i.test(t.headers.join(' ')));
  if (tables.length) {
    tables.sort((a, b) => b.rows.length - a.rows.length);
    return normalizeSizeChart('29cm', tables[0]);
  }

  throw new Error('29cm 페이지에서 사이즈 표를 찾을 수 없습니다.');
}

// ─────────────────────────────────────────────────────────────
// 일반 HTML 테이블 파서
// ─────────────────────────────────────────────────────────────
function parseHtmlTable($, tbl) {
  const headers = [];
  $(tbl).find('th').each((_, th) => headers.push($(th).text().trim()));
  if (headers.length === 0) {
    $(tbl).find('thead td, thead th').each((_, th) => headers.push($(th).text().trim()));
  }
  const rows = [];
  $(tbl).find('tr').each((_, tr) => {
    const cells = [];
    $(tr).find('td').each((_, td) => cells.push($(td).text().trim()));
    if (cells.length) rows.push(cells);
  });
  return { headers, rows };
}

// ─────────────────────────────────────────────────────────────
// 다양한 표 형식을 표준 형식으로 정규화
// 출력 형식:
//   {
//     mall: 'musinsa',
//     sizes: [
//       { label: 'M', measurements: { shoulder: 50.5, chest: 55.5, ... } }
//     ]
//   }
// ─────────────────────────────────────────────────────────────
function normalizeSizeChart(mall, raw) {
  // Case 1: 이미 정규화된 객체 배열
  if (Array.isArray(raw) && raw.length && typeof raw[0] === 'object' && !Array.isArray(raw[0])) {
    const sizes = raw.map(row => {
      const label = row.sizeNm || row.sizeName || row.label || row.size || row.optionName || '';
      const measurements = {};
      for (const [k, v] of Object.entries(row)) {
        if (['sizeNm', 'sizeName', 'label', 'size', 'optionName'].includes(k)) continue;
        const num = parseFloat(String(v).replace(/[^\d.]/g, ''));
        if (!isNaN(num) && num > 0 && num < 300) {
          const key = mapKey(k);
          if (key) measurements[key] = num;
        }
      }
      return { label: String(label).trim(), measurements };
    }).filter(s => s.label && Object.keys(s.measurements).length);
    return { mall, sizes };
  }

  // Case 2: HTML 표 (headers, rows)
  if (raw && Array.isArray(raw.headers) && Array.isArray(raw.rows)) {
    // headers: ['사이즈', '어깨너비', '가슴단면', '소매길이', '총장']
    const colMap = raw.headers.map(mapKey);
    const sizes = raw.rows.map(row => {
      const label = String(row[0] || '').trim();
      const measurements = {};
      row.forEach((cell, i) => {
        if (i === 0) return;
        const num = parseFloat(String(cell).replace(/[^\d.]/g, ''));
        const key = colMap[i];
        if (key && !isNaN(num) && num > 0 && num < 300) {
          measurements[key] = num;
        }
      });
      return { label, measurements };
    }).filter(s => s.label && Object.keys(s.measurements).length);
    return { mall, sizes };
  }

  throw new Error('사이즈 표 형식을 인식할 수 없습니다.');
}

const KEY_MAP = {
  '어깨': 'shoulder', '어깨너비': 'shoulder', 'shoulder': 'shoulder', 'shoulderWidth': 'shoulder',
  '가슴': 'chest', '가슴단면': 'chest', '가슴둘레': 'chest', 'chest': 'chest', 'chestWidth': 'chest', 'bust': 'chest',
  '소매': 'sleeve', '소매길이': 'sleeve', 'sleeve': 'sleeve', 'sleeveLength': 'sleeve',
  '총장': 'length', '총길이': 'length', 'length': 'length', 'totalLength': 'length', 'topLength': 'length',
  '허리': 'waist', 'waist': 'waist', 'waistWidth': 'waist',
  '힙': 'hip', '엉덩이': 'hip', 'hip': 'hip', 'hipWidth': 'hip',
  '허벅지': 'thigh', 'thigh': 'thigh', 'thighWidth': 'thigh',
  '밑위': 'rise', 'rise': 'rise', 'frontRise': 'rise',
  '허리단면': 'waist',
  '가슴폭': 'chest',
  '밑단': 'hem', 'hem': 'hem',
};

function mapKey(k) {
  if (!k) return null;
  const lower = String(k).toLowerCase().replace(/\s+/g, '');
  for (const [from, to] of Object.entries(KEY_MAP)) {
    if (lower.includes(from.toLowerCase())) return to;
  }
  return null;
}

// ─────────────────────────────────────────────────────────────
// 사이즈 추천
// userBody: { height, weight, shoulderWidth, chest, waist, hip }  (cm)
// sizeChart: normalizeSizeChart의 결과
// garmentType: 'upper' | 'lower' | 'outer' | 'full'
// ─────────────────────────────────────────────────────────────
function recommendSize(userBody, sizeChart, garmentType = 'upper') {
  if (!sizeChart || !sizeChart.sizes || !sizeChart.sizes.length) {
    return { ok: false, error: '사이즈 표가 비어있습니다.' };
  }

  // 비교에 사용할 신체 부위 결정
  const compareKeys = {
    upper: ['shoulder', 'chest'],
    outer: ['shoulder', 'chest'],
    full:  ['shoulder', 'chest', 'hip'],
    lower: ['waist', 'hip', 'thigh'],
  }[garmentType] || ['shoulder', 'chest'];

  // 가중치 (어깨가 상의 핏을 가장 크게 좌우)
  const weights = { shoulder: 0.6, chest: 0.4, waist: 0.4, hip: 0.3, thigh: 0.2, length: 0.1 };

  let best = null;
  let bestScore = Infinity;
  const ranked = [];

  for (const size of sizeChart.sizes) {
    let diff = 0;
    let used = 0;
    for (const k of compareKeys) {
      const m = size.measurements[k];
      const u = userBody[k];
      if (typeof m === 'number' && typeof u === 'number') {
        // +1cm 차이는 1점, 음수는 옷이 더 작음(빡빡)
        diff += weights[k] * Math.abs(m - u);
        used += weights[k];
      }
    }
    if (used === 0) continue;
    const score = diff / used;
    ranked.push({ label: size.label, score, measurements: size.measurements });
    if (score < bestScore) {
      bestScore = score;
      best = size;
    }
  }

  if (!best) {
    return { ok: false, error: '사용자 신체 정보와 매칭 가능한 사이즈가 없습니다.' };
  }

  ranked.sort((a, b) => a.score - b.score);

  // 추천 사이즈 외에 "한 단계 작게/크게"도 함께 제공
  const idx = ranked.findIndex(r => r.label === best.label);
  const smaller = idx + 1 < ranked.length ? ranked[idx + 1] : null;
  const larger  = idx > 0 ? ranked[idx - 1] : null;

  // 핏 라벨 (마진 기준)
  const margin = computeMargin(userBody, best.measurements, compareKeys);
  let fitLabel = '정사이즈';
  let fitDetail = '딱 맞게 떨어지는 핏입니다.';
  if (margin < -1.5) {
    fitLabel = '슬림핏 권장';
    fitDetail = `어깨/가슴이 평균 ${Math.abs(margin).toFixed(1)}cm 큽니다. 한 사이즈 업을 권장합니다.`;
  } else if (margin > 2.5) {
    fitLabel = '루즈핏';
    fitDetail = `평균 ${margin.toFixed(1)}cm 여유가 있습니다. 오버핏 느낌으로 착용됩니다.`;
  } else if (margin > 0.5) {
    fitLabel = '여유핏';
    fitDetail = `평균 ${margin.toFixed(1)}cm 여유. 약간 루즈하게 떨어집니다.`;
  }

  return {
    ok: true,
    recommended: {
      label: best.label,
      measurements: best.measurements,
      fitLabel,
      fitDetail,
      score: Number(bestScore.toFixed(2)),
    },
    alternatives: {
      smaller: smaller ? { label: smaller.label, measurements: smaller.measurements } : null,
      larger:  larger  ? { label: larger.label,  measurements: larger.measurements  } : null,
    },
    ranked: ranked.slice(0, 5).map(r => ({ label: r.label, score: Number(r.score.toFixed(2)) })),
  };
}

function computeMargin(userBody, sizeM, keys) {
  const vals = [];
  for (const k of keys) {
    const m = sizeM[k];
    const u = userBody[k];
    if (typeof m === 'number' && typeof u === 'number') vals.push(m - u);
  }
  if (!vals.length) return 0;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

// ─────────────────────────────────────────────────────────────
// 메인: URL과 userBody를 받아 사이즈 추천까지 한 번에
// ─────────────────────────────────────────────────────────────
async function fetchAndRecommend(url, userBody, garmentType) {
  const mall = detectMall(url);
  if (!mall) {
    throw new Error('지원하지 않는 쇼핑몰입니다. (무신사/지그재그/29cm)');
  }
  let chart;
  if (mall === 'musinsa') chart = await scrapeMusinsaSize(url);
  else if (mall === 'zigzag') chart = await scrapeZigzagSize(url);
  else if (mall === '29cm')  chart = await scrape29cmSize(url);

  const rec = recommendSize(userBody, chart, garmentType);
  return { mall, sizeChart: chart, recommendation: rec };
}

module.exports = {
  detectMall,
  scrapeMusinsaSize,
  scrapeZigzagSize,
  scrape29cmSize,
  recommendSize,
  fetchAndRecommend,
};
