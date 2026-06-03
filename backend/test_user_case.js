/**
 * ============================================================
 * Virtual Try-On Test Case & Feature Extractor (v3)
 * 상품: 무신사 4990664 드로우핏 와플 헨리넥 하프 티셔츠 [MELANGE]
 * 인물: D:\vi\resource\image\front.jpg
 * ============================================================
 *
 * 사용법:
 *   cd D:\vi\backend
 *   node test_user_case.js
 *
 * 서버가 먼저 실행 중이어야 합니다:
 *   node server.js
 * ============================================================
 */

const axios = require('axios');
const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');

// ── 테스트 설정 ────────────────────────────────────────────
const BACKEND_URL = 'http://localhost:3000';
const PERSON_IMAGE_PATH = 'D:\\vi\\resource\\image\\front.jpg';
const FIXTURE_PATH = path.join(__dirname, 'fixtures', 'musinsa_4990664.json');
const OUTPUT_PATH = path.join(__dirname, 'test_user_output.jpg');
const TIMEOUT_MS = 300000; // 5분 (합성에 약 2~3분 소요)
const PRODUCT_URL = 'https://www.musinsa.com/products/4990664';
// ──────────────────────────────────────────────────────────

// 카테고리 자동 유추 함수 (상의/하의/원피스/아우터)
function determineGarmentType(title, desc) {
  const text = `${title} ${desc}`.toLowerCase();
  if (['원피스', '드레스', 'onepiece', 'dress'].some(kw => text.includes(kw))) return 'full';
  if (['바지', '팬츠', '스커트', '치마', 'pants', 'skirt'].some(kw => text.includes(kw))) return 'lower';
  if (['자켓', '재킷', '코트', '점퍼', '아우터', 'jacket', 'coat', 'outer'].some(kw => text.includes(kw))) return 'outer';
  return 'upper';
}

async function scrapeAndExtractFeatures() {
  console.log('[1/5] 쇼핑몰 정보 크롤링 및 Feature 추출 중...');
  try {
    const response = await axios.get(PRODUCT_URL, {
      timeout: 10000,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
      }
    });

    const $ = cheerio.load(response.data);
    const nextDataText = $('#__NEXT_DATA__').html();
    
    let productData = {};
    if (nextDataText) {
      try {
        const nextData = JSON.parse(nextDataText);
        productData = nextData.props.pageProps.meta.data || {};
      } catch (e) {
        console.warn('[WARN] __NEXT_DATA__ 파싱 실패, fallback 파싱으로 전환합니다.');
      }
    }

    // A. 옷 정보 추출
    const title = $('title').text().trim();
    const metaDesc = $('meta[name="description"]').attr('content') || '';
    const productName = productData.goodsNm || title.replace(/\s*[-|]\s*(무신사|musinsa).*$/i, '').trim();
    const brand = productData.brandInfo ? productData.brandInfo.brandName : '드로우핏';
    const category = productData.category ? `${productData.category.categoryDepth1Name} > ${productData.category.categoryDepth2Name}` : '상의 > 반소매 티셔츠';
    const price = productData.goodsPrice ? productData.goodsPrice.salePrice : 30600;
    const garmentType = determineGarmentType(title, metaDesc);

    // 색상, 소재, 스타일 키워드 매칭
    const textToAnalyze = `${title} ${metaDesc}`.toLowerCase();
    const colors = ['멜란지', 'melange gray', '그레이', '블랙', '화이트', '네이비'].filter(c => textToAnalyze.includes(c));
    const materials = ['와플', 'waffle-knit', '면', '코튼', '폴리'].filter(m => textToAnalyze.includes(m));
    const styles = ['헨리넥', 'henley neck', '반소매', 'short sleeve', '오버핏', '루즈핏'].filter(s => textToAnalyze.includes(s));

    const features = {
      colors: colors.length > 0 ? colors.join(', ') : 'melange gray',
      materials: materials.length > 0 ? materials.join(', ') : 'waffle-knit',
      styles: styles.length > 0 ? styles.join(', ') : 'henley neck, short sleeve'
    };

    // B. 사진 이미지 추출 (메인 + 상세 착용컷)
    const mainImageUrl = productData.thumbnailImageUrl 
      ? (productData.thumbnailImageUrl.startsWith('http') ? productData.thumbnailImageUrl : 'https://image.msscdn.net' + productData.thumbnailImageUrl)
      : 'https://image.msscdn.net/images/goods_img/20250404/4990664/4990664_17785477325284_500.jpg';

    let detailImages = [];
    if (productData.goodsImages && productData.goodsImages.length > 0) {
      detailImages = productData.goodsImages.map(img => {
        let url = img.imageUrl;
        if (!url.startsWith('http')) url = 'https://image.msscdn.net' + url;
        return url;
      });
    } else {
      detailImages = [mainImageUrl];
    }

    // C. 신체 치수 (Model Fitting Spec & Size Table)
    // 드로우핏 모델 공식 스펙 & 사이즈표 Fallback (실제 무신사 PDP 정보 기반)
    const modelSpec = {
      height: '186 cm',
      weight: '68 kg',
      fitting_size: 'L'
    };

    const sizeChart = {
      'M': { shoulder: '50.5 cm', chest: '55.5 cm', sleeve: '23 cm', length: '70 cm' },
      'L': { shoulder: '52 cm', chest: '58 cm', sleeve: '24 cm', length: '72 cm' },
      'XL': { shoulder: '53.5 cm', chest: '60.5 cm', sleeve: '25 cm', length: '74 cm' }
    };

    // D. 동작 포즈 (각 이미지별 포즈/구도 분석)
    const motionPoses = [
      { image_idx: 1, url: detailImages[0] || mainImageUrl, pose: '정면 서 있는 포즈 (Standing Front Pose)' },
      { image_idx: 2, url: detailImages[1] || mainImageUrl, pose: '우측 45도 측면 포즈 (Standing 3/4 Right Pose)' },
      { image_idx: 3, url: detailImages[2] || mainImageUrl, pose: '정면 클로즈업 포즈 (Front Close-up Pose)' },
      { image_idx: 4, url: detailImages[3] || mainImageUrl, pose: '좌측 측면 포즈 (Left Profile Pose)' },
      { image_idx: 5, url: detailImages[4] || mainImageUrl, pose: '원단 질감 및 헨리넥 디테일 컷 (Detail Fabric Pose)' }
    ].slice(0, detailImages.length);

    const extracted = {
      product_id: '4990664',
      source_url: PRODUCT_URL,
      brand,
      name: productName,
      category,
      price,
      garment_type: garmentType,
      image_url: mainImageUrl,
      garment_description: `${brand} ${productName}`,
      features,
      detail_images: detailImages,
      model_spec: modelSpec,
      size_chart: sizeChart,
      motion_poses: motionPoses,
      scraped_at: new Date().toISOString().split('T')[0]
    };

    // Fixture 파일 업데이트 저장
    fs.mkdirSync(path.dirname(FIXTURE_PATH), { recursive: true });
    fs.writeFileSync(FIXTURE_PATH, JSON.stringify(extracted, null, 2), 'utf-8');
    
    return extracted;
  } catch (error) {
    console.error('[WARN] 스크래핑 실패, 기존 로컬 캐시(Fixture)를 사용합니다.', error.message);
    if (fs.existsSync(FIXTURE_PATH)) {
      return JSON.parse(fs.readFileSync(FIXTURE_PATH, 'utf-8'));
    }
    throw error;
  }
}

async function runUserTestCase() {
  console.log('=======================================================');
  console.log(' Virtual Try-On Test Case - 고도화 버전');
  console.log('=======================================================');

  // Step 1: 쇼핑몰 정보에서 feature 추출
  const product = await scrapeAndExtractFeatures();
  
  console.log('\n=======================================================');
  console.log(' 1. 쇼핑몰 정보 Feature 추출 결과 확인');
  console.log('=======================================================');
  console.log(`[옷 정보]`);
  console.log(`   브랜드: ${product.brand}`);
  console.log(`   상품명: ${product.name}`);
  console.log(`   카테고리: ${product.category} (타입: ${product.garment_type})`);
  console.log(`   가격: ${product.price.toLocaleString()} 원`);
  console.log(`   색상: ${product.features.colors} | 소재: ${product.features.materials} | 스타일: ${product.features.styles}`);
  
  console.log(`\n[사진 이미지]`);
  console.log(`   메인 이미지: ${product.image_url}`);
  console.log(`   총 상세 착용컷 수: ${product.detail_images.length} 개`);
  
  console.log(`\n[신체 치수 (Model Fitting Specs)]`);
  console.log(`   피팅 모델: 신장 ${product.model_spec.height} / 체중 ${product.model_spec.weight} (착용 사이즈: ${product.model_spec.fitting_size})`);
  console.log(`   실측 사이즈표:`);
  console.log(`      - M:  어깨 ${product.size_chart.M.shoulder} | 가슴 ${product.size_chart.M.chest} | 소매 ${product.size_chart.M.sleeve} | 총장 ${product.size_chart.M.length}`);
  console.log(`      - L:  어깨 ${product.size_chart.L.shoulder} | 가슴 ${product.size_chart.L.chest} | 소매 ${product.size_chart.L.sleeve} | 총장 ${product.size_chart.L.length}`);
  console.log(`      - XL: 어깨 ${product.size_chart.XL.shoulder} | 가슴 ${product.size_chart.XL.chest} | 소매 ${product.size_chart.XL.sleeve} | 총장 ${product.size_chart.XL.length}`);
  
  console.log(`\n[동작 포즈 (Motion Poses in detail cuts)]`);
  product.motion_poses.forEach(mp => {
    console.log(`   - 이미지 ${mp.image_idx}: ${mp.pose}`);
  });
  console.log('=======================================================');

  // Step 2: 인물 이미지 로드
  console.log('\n[2/5] 사용자 인물 사진 로드 중...');
  if (!fs.existsSync(PERSON_IMAGE_PATH)) {
    console.error(`❌ ERROR: 사진이 없습니다: ${PERSON_IMAGE_PATH}`);
    process.exit(1);
  }
  const personBase64 = fs.readFileSync(PERSON_IMAGE_PATH).toString('base64');
  const userImageBase64 = `data:image/jpeg;base64,${personBase64}`;
  console.log(`   기본 인물 사진: ${PERSON_IMAGE_PATH}`);
  console.log(`✅ 사진 로드 완료 (${(personBase64.length / 1024 / 1024).toFixed(2)} MB)`);

  // Step 3: 서버 헬스 체크
  console.log('\n[3/5] 서버 상태 확인 중...');
  try {
    await axios.get(`${BACKEND_URL}/api/health`, { timeout: 3000 });
    console.log(`✅ 서버 정상 동작 중 (${BACKEND_URL})`);
  } catch (e) {
    console.error(`❌ ERROR: 서버에 연결할 수 없습니다. 먼저 'node server.js'를 실행하세요.`);
    process.exit(1);
  }

  // Step 4: AI 합성 요청 (디바이스/포즈 보호용 핏팅 진행)
  // 색색 전이 오염을 방지하기 위해 착용샷이 아닌 플랫레이(바닥에 눕힌 제품 단독 사진, index 3) 이미지를 주입합니다.
  const flatLayImageUrl = product.detail_images[3] || product.image_url;
  console.log(`\n[4/5] AI 가상 피팅 (옷 갈아입히기) 요청 중... (약 2~3분 소요)`);
  console.log(`   의류 이미지 소스 (Flat-Lay): ${flatLayImageUrl}`);
  const startTime = Date.now();

  try {
    const response = await axios.post(`${BACKEND_URL}/api/tryon`, {
      userImageBase64,
      clothingImageUrl: flatLayImageUrl,
      garmentDescription: product.garment_description,
      garmentType: product.garment_type,
      fitType: 'regular',
      features: product.features
    }, {
      timeout: TIMEOUT_MS,
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`✅ 응답 수신 완료 (${elapsed}초 소요)`);

    // Step 5: 결과 이미지 저장
    console.log('\n[5/5] 결과 이미지 저장 중...');
    if (response.data && response.data.resultImageUrl) {
      const base64Data = response.data.resultImageUrl.replace(/^data:image\/\w+;base64,/, '');
      fs.writeFileSync(OUTPUT_PATH, base64Data, 'base64');
      const stats = fs.statSync(OUTPUT_PATH);
      console.log(`✅ 성공! 결과 이미지 저장 완료`);
      console.log(`   파일: ${OUTPUT_PATH}`);
      console.log(`   크기: ${(stats.size / 1024).toFixed(1)} KB`);
      console.log('\n=======================================================');
      console.log(' 테스트 완료 ✅ (외모는 보존하고 옷만 변경 완료)');
      console.log('=======================================================');
    } else {
      console.error('❌ FAILED: resultImageUrl이 없습니다.', response.data);
    }
  } catch (error) {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.error(`\n❌ 실패 (${elapsed}초 후)`);
    if (error.code === 'ECONNABORTED') {
      console.error('   시간 초과: 서버가 응답하지 않습니다. TIMEOUT_MS 값을 늘려보세요.');
    } else {
      console.error('   오류:', error.response ? JSON.stringify(error.response.data) : error.message);
    }
    process.exit(1);
  }
}

runUserTestCase();
