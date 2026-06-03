function determineGarmentType(title, desc) {
  const text = `${title} ${desc}`.toLowerCase();
  const fullKeywords = ['원피스', '드레스', '점프수트', '점프슈트', 'onepiece', 'one-piece', 'dress', 'jumpsuit'];
  if (fullKeywords.some(kw => text.includes(kw))) return 'full';
  const lowerKeywords = ['바지', '팬츠', '데님', '청바지', '스커트', '치마', '슬랙스', '레깅스', '반바지', '숏팬츠', 'pants', 'skirt', 'jeans', 'denim', 'slacks', 'shorts', 'trouser', 'trousers'];
  if (lowerKeywords.some(kw => text.includes(kw))) return 'lower';
  const outerKeywords = ['자켓', '재킷', '코트', '가디건', '패딩', '점퍼', '아우터', '가죽', '라이더', '블레이저', '무스탕', 'jacket', 'coat', 'cardigan', 'jumper', 'outer', 'blazer', 'parka', 'windbreaker'];
  if (outerKeywords.some(kw => text.includes(kw))) return 'outer';
  return 'upper';
}

function extractFeatures(title, desc) {
  const text = `${title} ${desc}`.toLowerCase();

  // Color keywords (Korean + English)
  const colorMap = {
    '블랙': 'black', '화이트': 'white', '네이비': 'navy blue', '그레이': 'gray',
    '베이지': 'beige', '멜란지': 'melange gray', '아이보리': 'ivory', '카키': 'khaki',
    '브라운': 'brown', '레드': 'red', '블루': 'blue', '그린': 'green',
    '옐로우': 'yellow', '핑크': 'pink', '퍼플': 'purple', '오렌지': 'orange',
    'black': 'black', 'white': 'white', 'navy': 'navy blue', 'gray': 'gray',
    'grey': 'gray', 'beige': 'beige', 'melange': 'melange gray', 'ivory': 'ivory',
    'khaki': 'khaki', 'brown': 'brown', 'blue': 'blue', 'green': 'green',
  };

  // Material keywords
  const materialMap = {
    '면': 'cotton', '코튼': 'cotton', '린넨': 'linen', '니트': 'knit', '울': 'wool',
    '폴리': 'polyester', '와플': 'waffle-knit', '캐시미어': 'cashmere',
    '스웨이드': 'suede', '데님': 'denim', '가죽': 'leather', '시폰': 'chiffon',
    'cotton': 'cotton', 'linen': 'linen', 'knit': 'knit', 'wool': 'wool',
    'waffle': 'waffle-knit', 'denim': 'denim', 'leather': 'leather',
    'polyester': 'polyester', 'cashmere': 'cashmere',
  };

  // Style keywords
  const styleMap = {
    '오버핏': 'oversized', '오버사이즈': 'oversized', '슬림': 'slim fit',
    '루즈': 'loose fit', '크롭': 'cropped', '박시': 'boxy', '스트릿': 'streetwear',
    '캐주얼': 'casual', '헨리넥': 'henley neck', '브이넥': 'v-neck',
    '라운드넥': 'round neck', '터틀넥': 'turtleneck', '후드': 'hoodie',
    '반소매': 'short sleeve', '긴소매': 'long sleeve', '민소매': 'sleeveless',
    'oversize': 'oversized', 'slim': 'slim fit', 'crop': 'cropped',
    'boxy': 'boxy', 'henley': 'henley neck', 'hoodie': 'hoodie',
  };

  const foundColors = [];
  const foundMaterials = [];
  const foundStyles = [];

  for (const [kw, val] of Object.entries(colorMap)) {
    if (text.includes(kw) && !foundColors.includes(val)) foundColors.push(val);
  }
  for (const [kw, val] of Object.entries(materialMap)) {
    if (text.includes(kw) && !foundMaterials.includes(val)) foundMaterials.push(val);
  }
  for (const [kw, val] of Object.entries(styleMap)) {
    if (text.includes(kw) && !foundStyles.includes(val)) foundStyles.push(val);
  }

  return {
    colors: foundColors.join(', '),
    materials: foundMaterials.join(', '),
    styles: foundStyles.join(', '),
  };
}

module.exports = {
  determineGarmentType,
  extractFeatures
};
