const test = require('node:test');
const assert = require('node:assert');
const { determineGarmentType, extractFeatures } = require('./helpers');

test('determineGarmentType categorization tests', () => {
  // Upper body tests
  assert.strictEqual(determineGarmentType('와플 헨리넥 하프 티셔츠', '반소매 상의 티셔츠'), 'upper');
  assert.strictEqual(determineGarmentType('T-shirt', 'standard daily cotton tee'), 'upper');

  // Lower body tests
  assert.strictEqual(determineGarmentType('드로우핏 데님 팬츠 [BLUE]', '남자 와이드 청바지 바지'), 'lower');
  assert.strictEqual(determineGarmentType('A-line Skirt', '플레어 스커트 치마'), 'lower');

  // Full body / Dress tests
  assert.strictEqual(determineGarmentType('플라워 쉬폰 원피스', '롱 드레스 드레스'), 'full');
  assert.strictEqual(determineGarmentType('Denim Jumpsuit', '점프수트'), 'full');

  // Outerwear tests
  assert.strictEqual(determineGarmentType('오버핏 가죽 자켓', '겨울 무스탕 코트 재킷'), 'outer');
  assert.strictEqual(determineGarmentType('Winter Parka', '따뜻한 패딩 점퍼 아우터'), 'outer');
});

test('extractFeatures visual features extraction tests', () => {
  // Test color, material and style extraction
  const f1 = extractFeatures('와플 헨리넥 하프 티셔츠 [MELANGE]', '드로우핏 면 코튼 루즈핏 오버사이즈 반소매');
  assert.ok(f1.colors.includes('melange gray'));
  assert.ok(f1.materials.includes('waffle-knit') && f1.materials.includes('cotton'));
  assert.ok(f1.styles.includes('henley neck') && f1.styles.includes('oversized') && f1.styles.includes('short sleeve'));

  // Test default/empty extraction
  const f2 = extractFeatures('무지 셔츠', '단순하고 깔끔한 디자인');
  assert.strictEqual(f2.colors, '');
  assert.strictEqual(f2.materials, '');
  assert.strictEqual(f2.styles, '');
});
