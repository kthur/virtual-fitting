import React, { useState, useCallback, useEffect, useMemo, memo, useRef } from 'react';
import {
  StyleSheet, Text, View, TextInput, TouchableOpacity,
  Image, ActivityIndicator, Alert, SafeAreaView, ScrollView,
  KeyboardAvoidingView, Platform, FlatList
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Image as ExpoImage, prefetch } from 'expo-image';
import axios from 'axios';
import { preprocessUserPhoto, assessPhotoQuality } from './src/vision/imagePreprocess';
import { parseCameraCalibration, computeCalibration } from './src/vision/cameraCalibrate';

const BACKEND_URL = 'http://192.168.31.231:3000';

const STEPS = {
  SELECT_PHOTO: 'SELECT_PHOTO',
  CAMERA:       'CAMERA',
  INPUT_BODY:   'INPUT_BODY',
  ESTIMATING:   'ESTIMATING',
  INPUT_URL:    'INPUT_URL',
  SCRAPING:     'SCRAPING',
  SIZE_RESULT:  'SIZE_RESULT',
  IMAGE_SELECTION: 'IMAGE_SELECTION',
  AI_FITTING:   'AI_FITTING',
  RESULT:       'RESULT',
};

export default function App() {
  const [step, setStep] = useState(STEPS.SELECT_PHOTO);

  // 사용자 사진
  const [userPhoto, setUserPhoto] = useState(null);
  const [userPhotoBase64, setUserPhotoBase64] = useState('');
  const [photoQuality, setPhotoQuality] = useState(null);

  // 사용자 신체 정보 (cm, kg)
  const [userBody, setUserBody] = useState({
    height: '', weight: '',
    shoulderWidth: '', chest: '', waist: '', hip: '', thigh: '',
  });
  const [bodyEstimated, setBodyEstimated] = useState(false);

  // 쇼핑몰 URL / 스크래핑 결과
  const [shoppingUrl, setShoppingUrl] = useState('');
  const [scrapedImages, setScrapedImages] = useState([]);
  const [selectedClothingImage, setSelectedClothingImage] = useState(null);
  const [garmentDescription, setGarmentDescription] = useState('');
  const [garmentType, setGarmentType] = useState('upper');
  const [fitType, setFitType] = useState('regular');
  const [features, setFeatures] = useState({});
  const [sizeChart, setSizeChart] = useState(null);
  const [sizeRecommendation, setSizeRecommendation] = useState(null);

  // AI 합성
  const [finalImage, setFinalImage] = useState(null);
  const [useUpscale, setUseUpscale] = useState(false);

  // 카메라
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [cameraCalibration, setCameraCalibration] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);
  const cameraRef = useRef(null);

  // ───────── 사진 선택 (EXIF 보정 + 리사이즈) ─────────
  const pickImage = useCallback(async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('권한 필요', '사진 라이브러리 접근 권한이 필요합니다.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 0.7,
      base64: true,
    });
    if (!result.canceled) {
      try {
        const processed = await preprocessUserPhoto(result.assets[0].uri);
        setUserPhoto(processed.uri);
        setUserPhotoBase64(processed.dataUri);
        setPhotoQuality(processed.quality);

        if (processed.quality.issues.length) {
          const hard = processed.quality.issues.find(i => i.severity === 'warn');
          if (hard) {
            Alert.alert('사진 품질 안내', hard.message, [{ text: '계속' }]);
          }
        }
        setStep(STEPS.INPUT_BODY);
      } catch (e) {
        Alert.alert('사진 처리 실패', e.message || '이미지를 정규화하지 못했습니다.');
      }
    }
  }, []);

  // ───────── 카메라 촬영 ─────────
  const takeCameraPhoto = useCallback(async () => {
    if (!cameraPermission?.granted) {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('권한 필요', '카메라 접근 권한이 필요합니다.');
        return;
      }
    }
    setStep(STEPS.CAMERA);
  }, [cameraPermission]);

  const handleCameraCapture = useCallback(async () => {
    if (!cameraRef.current) return;
    setCameraReady(false);
    try {
      const photo = await cameraRef.current.takePictureAsync({ base64: true, quality: 0.7 });
      if (!photo?.uri) { Alert.alert('오류', '사진을 찍지 못했습니다.'); return; }

      const cal = await parseCameraCalibration(photo.uri);
      setCameraCalibration(cal.available ? cal : null);

      const processed = await preprocessUserPhoto(photo.uri);
      setUserPhoto(processed.uri);
      setUserPhotoBase64(processed.dataUri);
      setPhotoQuality(processed.quality);

      if (processed.quality.issues.length) {
        const hard = processed.quality.issues.find(i => i.severity === 'warn');
        if (hard) Alert.alert('사진 품질 안내', hard.message, [{ text: '계속' }]);
      }
      setStep(STEPS.INPUT_BODY);
    } catch (e) {
      Alert.alert('사진 처리 실패', e.message || '카메라 사진을 처리하지 못했습니다.');
    } finally {
      setCameraReady(true);
    }
  }, []);

  // Prefetch scraped images as soon as they're available so the carousel renders instantly
  useEffect(() => {
    if (scrapedImages.length) {
      prefetch(scrapedImages, 'memory-disk').catch(() => {});
    }
  }, [scrapedImages]);

  // ───────── 신체 자동 추정 (사진 + 키 + EXIF 보정) ─────────
  const estimateBody = useCallback(async () => {
    if (!userPhotoBase64) return;
    setStep(STEPS.ESTIMATING);

    let camera = null;
    if (cameraCalibration?.available) {
      camera = computeCalibration(
        cameraCalibration,
        photoQuality?.width || 0,
        photoQuality?.height || 0,
        0,
        userBody.height ? parseFloat(userBody.height) : null,
        true,
      );
    }

    try {
      const res = await axios.post(`${BACKEND_URL}/api/estimate-body`, {
        userImageBase64: userPhotoBase64,
        knownHeightCm: userBody.height ? parseFloat(userBody.height) : null,
        cameraCalibration: camera,
      }, { timeout: 60000, headers: { 'Bypass-Tunnel-Reminder': 'true' } });

      if (!res.data.ok) {
        Alert.alert('자동 추정 실패', res.data.error || '신체 랜드마크를 찾지 못했습니다.');
        setStep(STEPS.INPUT_BODY);
        return;
      }

      const cm = res.data.measurementsCm;
      if (cm) {
        setUserBody(prev => ({
          ...prev,
          shoulderWidth: cm.shoulderWidth ? String(Math.round(cm.shoulderWidth * 10) / 10) : prev.shoulderWidth,
          chest:         cm.chest ? String(Math.round(cm.chest * 10) / 10)         : prev.chest,
          waist:         cm.waist  != null ? String(Math.round(cm.waist * 10) / 10) : prev.waist,
          hip:           cm.hipWidth != null ? String(Math.round(cm.hipWidth * 10) / 10) : prev.hip,
        }));
        setBodyEstimated(true);
      } else {
        Alert.alert(
          '부분 추정 완료',
          '키를 입력하면 cm 단위로 환산할 수 있습니다. 비율은 추출되었지만, 지금은 수동으로 입력해 주세요.'
        );
      }
      setStep(STEPS.INPUT_BODY);
    } catch (e) {
      Alert.alert('오류', '신체 추정 실패: ' + (e.response?.data?.error || e.message));
      setStep(STEPS.INPUT_BODY);
    }
  }, [userPhotoBase64, userBody.height, cameraCalibration, photoQuality]);

  // ───────── 쇼핑몰 스크래핑 + 사이즈 표 ─────────
  const scrapeClothing = useCallback(async () => {
    if (!shoppingUrl) { Alert.alert('오류', 'URL을 입력해주세요.'); return; }
    setStep(STEPS.SCRAPING);
    try {
      // 1. 일반 스크래핑 (이미지 + 특징)
      const scrapeRes = await axios.post(`${BACKEND_URL}/api/scrape`,
        { url: shoppingUrl },
        { timeout: 15000, headers: { 'Bypass-Tunnel-Reminder': 'true' } }
      );
      const { imageUrls, garmentDescription: d, garmentType: t, features: f } = scrapeRes.data;
      if (!imageUrls || imageUrls.length === 0) throw new Error('옷 이미지를 찾을 수 없습니다.');

      setScrapedImages(imageUrls);
      setSelectedClothingImage(imageUrls[0]);
      setGarmentDescription(d || '');
      setGarmentType(t || 'upper');
      setFeatures(f || {});

      // 2. 사이즈 표 가져오기 (실패해도 계속 진행)
      try {
        const sizeRes = await axios.post(`${BACKEND_URL}/api/scrape-size`, {
          url: shoppingUrl,
          garmentType: t || 'upper',
        }, { timeout: 20000, headers: { 'Bypass-Tunnel-Reminder': 'true' } });

        if (sizeRes.data && sizeRes.data.sizeChart) {
          setSizeChart(sizeRes.data.sizeChart);

          // 3. 신체 정보가 있으면 자동 추천
          const body = parseBodyForApi(userBody);
          if (body && Object.keys(body).length >= 1) {
            try {
              const recRes = await axios.post(`${BACKEND_URL}/api/recommend-size`, {
                url: shoppingUrl,
                userBody: body,
                garmentType: t || 'upper',
              }, { timeout: 20000, headers: { 'Bypass-Tunnel-Reminder': 'true' } });
              if (recRes.data && recRes.data.recommendation) {
                setSizeRecommendation(recRes.data.recommendation);
                setStep(STEPS.SIZE_RESULT);
                return;
              }
            } catch (_) { /* 추천 실패는 무시하고 진행 */ }
          }
        }
      } catch (e) {
        console.warn('Size scrape skipped:', e.response?.data?.error || e.message);
      }

      setFitType('regular');
      setStep(STEPS.IMAGE_SELECTION);
    } catch (error) {
      Alert.alert('오류', '스크래핑 실패: ' + (error.response?.data?.error || error.message));
      setStep(STEPS.INPUT_URL);
    }
  }, [shoppingUrl, userBody]);

  // ───────── AI 가상 피팅 ─────────
  const processVirtualTryOn = useCallback(async () => {
    setStep(STEPS.AI_FITTING);
    try {
      const tryonRes = await axios.post(`${BACKEND_URL}/api/tryon`, {
        userImageBase64: userPhotoBase64,
        clothingImageUrl: selectedClothingImage,
        garmentDescription,
        garmentType,
        fitType,
        features,
        ipScale: 0.55,
        inferenceSteps: 35,
        upscale: useUpscale,
      }, {
        timeout: 300000,
        headers: { 'Bypass-Tunnel-Reminder': 'true' },
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
      });
      if (tryonRes.data && tryonRes.data.resultImageUrl) {
        setFinalImage(tryonRes.data.resultImageUrl);
        setStep(STEPS.RESULT);
      } else {
        throw new Error('AI 합성 결과를 받지 못했습니다.');
      }
    } catch (error) {
      if (error.code === 'ECONNABORTED') {
        Alert.alert('시간 초과', 'AI 서버가 바쁩니다. 잠시 후 다시 시도해주세요.');
      } else {
        Alert.alert('오류', error.response?.data?.error || error.message);
      }
      setStep(STEPS.IMAGE_SELECTION);
    }
  }, [userPhotoBase64, selectedClothingImage, garmentDescription, garmentType, fitType, features, useUpscale]);

  // ───────── Reset helpers ─────────
  const resetAll = useCallback(() => {
    setShoppingUrl('');
    setScrapedImages([]);
    setSelectedClothingImage(null);
    setGarmentDescription('');
    setSizeChart(null);
    setSizeRecommendation(null);
    setFinalImage(null);
    setStep(STEPS.INPUT_URL);
  }, []);

  const startOver = useCallback(() => {
    setUserPhoto(null);
    setUserPhotoBase64('');
    setUserBody({ height: '', weight: '', shoulderWidth: '', chest: '', waist: '', hip: '', thigh: '' });
    setBodyEstimated(false);
    setFinalImage(null);
    setStep(STEPS.SELECT_PHOTO);
  }, []);

  // ───────── Body helpers ─────────
  const parseBodyForApi = useCallback((b) => {
    const out = {};
    if (b.height) out.height = parseFloat(b.height);
    if (b.weight) out.weight = parseFloat(b.weight);
    if (b.shoulderWidth) out.shoulder = parseFloat(b.shoulderWidth);
    if (b.chest)         out.chest   = parseFloat(b.chest);
    if (b.waist)         out.waist   = parseFloat(b.waist);
    if (b.hip)           out.hip     = parseFloat(b.hip);
    if (b.thigh)         out.thigh   = parseFloat(b.thigh);
    return out;
  }, []);

  const updateBody = useCallback((key, value) => {
    setUserBody(prev => ({ ...prev, [key]: value }));
  }, []);

  // ─────────────────────────────────────────────────────────
  //  RENDER
  // ─────────────────────────────────────────────────────────
  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>

        {/* ───── STEP 1: 사진 선택 ───── */}
        {step === STEPS.SELECT_PHOTO && (
          <View style={styles.centerContainer}>
            <Text style={styles.emoji}>👗</Text>
            <Text style={styles.title}>Virtual Fitting AI</Text>
            <Text style={styles.subtitle}>내 전신 또는 상반신 사진을 선택하면{'\n'}AI가 옷을 입혀드립니다</Text>
            <TouchableOpacity style={[styles.button, { marginBottom: 12 }]} onPress={takeCameraPhoto}>
              <Text style={styles.buttonText}>📷 카메라로 촬영</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.button, styles.secondaryButton]} onPress={pickImage}>
              <Text style={styles.buttonText}>🖼️ 갤러리에서 선택</Text>
            </TouchableOpacity>
            <Text style={styles.tip}>💡 카메라 촬영 시 EXIF 초점거리로 더 정확한 사이즈 추정</Text>
          </View>
        )}

        {/* ───── STEP 1.5: 카메라 뷰 ───── */}
        {step === STEPS.CAMERA && (
          <View style={{ flex: 1 }}>
            <CameraView
              ref={cameraRef}
              style={{ flex: 1 }}
              facing="front"
              onCameraReady={() => setCameraReady(true)}
            >
              <View style={{ flex: 1, justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 60 }}>
                <TouchableOpacity
                  style={{
                    width: 80, height: 80, borderRadius: 40,
                    backgroundColor: '#fff', borderWidth: 4, borderColor: '#4D96FF',
                    alignItems: 'center', justifyContent: 'center',
                  }}
                  onPress={handleCameraCapture}
                >
                  <View style={{ width: 60, height: 60, borderRadius: 30, backgroundColor: '#fff' }} />
                </TouchableOpacity>
              </View>
            </CameraView>
            <TouchableOpacity
              style={[styles.button, styles.secondaryButton, { position: 'absolute', top: 50, left: 20 }]}
              onPress={() => setStep(STEPS.SELECT_PHOTO)}
            >
              <Text style={styles.buttonText}>← 취소</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* ───── STEP 2: 신체 사이즈 입력 ───── */}
        {step === STEPS.INPUT_BODY && (
          <ScrollView contentContainerStyle={styles.scrollCenter} keyboardShouldPersistTaps="handled">
            <ExpoImage source={{ uri: userPhoto }} style={styles.thumbnail} />
            {photoQuality && (
              <Text style={styles.photoMeta}>
                📐 {photoQuality.width}×{photoQuality.height} px
                {cameraCalibration?.available && (
                  <Text> · 📷 초점 {cameraCalibration.focalLength35mm}mm
                    {cameraCalibration.isWideAngle && <Text style={{ color: '#FFB347' }}> (광각)</Text>}
                  </Text>
                )}
                {!cameraCalibration?.available && <Text> · EXIF 보정됨</Text>}
                {photoQuality.issues.length > 0 && (
                  <Text style={{ color: '#FFB347' }}> · {photoQuality.issues[0].message}</Text>
                )}
              </Text>
            )}
            <Text style={styles.title}>신체 사이즈를 알려주세요</Text>
            <Text style={styles.subtitle}>
              {bodyEstimated ? '✅ 사진으로 자동 추정했어요. 수정하고 싶은 부분만 바꿔주세요.' : '키를 입력하면 사진에서 자동으로 추정할 수 있어요.'}
            </Text>

            <View style={styles.bodyGrid}>
              <BodyField label="키 (cm) *"    value={userBody.height}        onChange={v => updateBody('height', v)}        placeholder="170" />
              <BodyField label="몸무게 (kg)"  value={userBody.weight}        onChange={v => updateBody('weight', v)}        placeholder="65" optional />
              <BodyField label="어깨너비"     value={userBody.shoulderWidth} onChange={v => updateBody('shoulderWidth', v)} placeholder="45"  optional />
              <BodyField label="가슴"          value={userBody.chest}         onChange={v => updateBody('chest', v)}         placeholder="95"  optional />
              <BodyField label="허리"          value={userBody.waist}         onChange={v => updateBody('waist', v)}         placeholder="75"  optional />
              <BodyField label="힙"            value={userBody.hip}           onChange={v => updateBody('hip', v)}           placeholder="95"  optional />
              <BodyField label="허벅지"        value={userBody.thigh}         onChange={v => updateBody('thigh', v)}         placeholder="55"  optional />
            </View>

            <View style={styles.buttonRow}>
              <TouchableOpacity
                style={[styles.button, styles.secondaryButton]}
                onPress={() => setStep(STEPS.SELECT_PHOTO)}
              >
                <Text style={styles.buttonText}>← 뒤로</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.button, styles.secondaryButton]}
                onPress={estimateBody}
                disabled={!userBody.height}
              >
                <Text style={styles.buttonText}>📏 사진으로 추정</Text>
              </TouchableOpacity>
            </View>

            <TouchableOpacity
              style={[styles.button, styles.primaryButton, { marginTop: 8 }]}
              onPress={() => setStep(STEPS.INPUT_URL)}
            >
              <Text style={styles.buttonText}>다음 → 쇼핑몰 URL</Text>
            </TouchableOpacity>

            <Text style={styles.tip}>* 키는 사진 추정과 사이즈 추천에 필요해요</Text>
          </ScrollView>
        )}

        {step === STEPS.ESTIMATING && (
          <View style={styles.centerContainer}>
            <ActivityIndicator size="large" color="#4D96FF" />
            <Text style={styles.loadingTitle}>신체 비율 분석 중...</Text>
            <Text style={styles.loadingSubtitle}>MediaPipe로 어깨·가슴·힙을 추정하고 있어요</Text>
          </View>
        )}

        {/* ───── STEP 3: URL 입력 ───── */}
        {step === STEPS.INPUT_URL && (
          <ScrollView contentContainerStyle={styles.scrollCenter} keyboardShouldPersistTaps="handled">
            <ExpoImage source={{ uri: userPhoto }} style={styles.thumbnail} />
            <Text style={styles.title}>어떤 옷을 입어볼까요?</Text>
            <Text style={styles.subtitle}>무신사 · 지그재그 · 29cm URL을 붙여넣으세요</Text>
            <TextInput
              style={styles.input}
              placeholder="https://www.musinsa.com/products/..."
              placeholderTextColor="#666"
              value={shoppingUrl}
              onChangeText={setShoppingUrl}
              autoCapitalize="none"
              autoCorrect={false}
            />
            <View style={styles.buttonRow}>
              <TouchableOpacity style={[styles.button, styles.secondaryButton]} onPress={() => setStep(STEPS.INPUT_BODY)}>
                <Text style={styles.buttonText}>← 뒤로</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.button} onPress={scrapeClothing}>
                <Text style={styles.buttonText}>옷 정보 가져오기 🔍</Text>
              </TouchableOpacity>
            </View>
          </ScrollView>
        )}

        {step === STEPS.SCRAPING && (
          <View style={styles.centerContainer}>
            <ActivityIndicator size="large" color="#4D96FF" />
            <Text style={styles.loadingTitle}>정보를 수집하는 중...</Text>
            <Text style={styles.loadingSubtitle}>쇼핑몰에서 옷 사진·상세 정보·사이즈 표를 찾고 있습니다</Text>
          </View>
        )}

        {/* ───── STEP 4: 사이즈 추천 결과 ───── */}
        {step === STEPS.SIZE_RESULT && sizeRecommendation && (
          <ScrollView contentContainerStyle={styles.scrollCenter} keyboardShouldPersistTaps="handled">
            <Text style={styles.emoji}>📐</Text>
            <Text style={styles.title}>나에게 맞는 사이즈</Text>

            <View style={styles.recommendCard}>
              <Text style={styles.recommendLabel}>추천 사이즈</Text>
              <Text style={styles.recommendSize}>{sizeRecommendation.label}</Text>
              <Text style={styles.recommendFit}>{sizeRecommendation.fitLabel}</Text>
              <Text style={styles.recommendDetail}>{sizeRecommendation.fitDetail}</Text>

              {sizeRecommendation.measurements && (
                <View style={styles.measureRow}>
                  {Object.entries(sizeRecommendation.measurements).map(([k, v]) => (
                    <View key={k} style={styles.measureChip}>
                      <Text style={styles.measureKey}>{MEASURE_LABEL[k] || k}</Text>
                      <Text style={styles.measureVal}>{v} cm</Text>
                    </View>
                  ))}
                </View>
              )}
            </View>

            {sizeRecommendation.alternatives && (sizeRecommendation.alternatives.smaller || sizeRecommendation.alternatives.larger) && (
              <View style={styles.altRow}>
                {sizeRecommendation.alternatives.smaller && (
                  <View style={styles.altCard}>
                    <Text style={styles.altLabel}>한 사이즈 작게</Text>
                    <Text style={styles.altSize}>{sizeRecommendation.alternatives.smaller.label}</Text>
                  </View>
                )}
                {sizeRecommendation.alternatives.larger && (
                  <View style={styles.altCard}>
                    <Text style={styles.altLabel}>한 사이즈 크게</Text>
                    <Text style={styles.altSize}>{sizeRecommendation.alternatives.larger.label}</Text>
                  </View>
                )}
              </View>
            )}

            {sizeChart && sizeChart.sizes && sizeChart.sizes.length > 0 && (
              <View style={styles.sizeTableBox}>
                <Text style={styles.sizeTableTitle}>전체 사이즈 표</Text>
                <SizeTable sizeChart={sizeChart} />
              </View>
            )}

            <View style={styles.buttonRow}>
              <TouchableOpacity style={[styles.button, styles.secondaryButton]} onPress={() => setStep(STEPS.INPUT_URL)}>
                <Text style={styles.buttonText}>← 다른 URL</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.button} onPress={() => setStep(STEPS.IMAGE_SELECTION)}>
                <Text style={styles.buttonText}>피팅하러 가기 →</Text>
              </TouchableOpacity>
            </View>
          </ScrollView>
        )}

        {/* ───── STEP 5: 옷 이미지 선택 ───── */}
        {step === STEPS.IMAGE_SELECTION && (
          <ScrollView contentContainerStyle={styles.scrollCenter} keyboardShouldPersistTaps="handled">
            <Text style={styles.title}>합성할 옷 이미지를 선택하세요</Text>
            <Text style={styles.subtitle}>누끼컷(배경 제거된 사진)이 가장 자연스러워요.</Text>

            <FlatList
              data={scrapedImages}
              keyExtractor={(uri, i) => `${i}:${uri}`}
              horizontal
              showsHorizontalScrollIndicator={false}
              style={styles.carousel}
              initialNumToRender={4}
              windowSize={5}
              removeClippedSubviews
              renderItem={({ item }) => (
                <TouchableOpacity onPress={() => setSelectedClothingImage(item)}>
                  <ExpoImage
                    source={{ uri: item }}
                    style={[styles.carouselImage, selectedClothingImage === item && styles.selectedImage]}
                    contentFit="cover"
                    cachePolicy="memory-disk"
                  />
                </TouchableOpacity>
              )}
            />

            <Text style={[styles.title, { marginTop: 24, fontSize: 20 }]}>옷 설명 (AI 프롬프트)</Text>
            <Text style={styles.subtitle}>추출된 정보입니다. 필요시 수정하세요.</Text>
            <TextInput
              style={[styles.input, { height: 80, textAlignVertical: 'top' }]}
              multiline
              placeholder="예: 오버핏 쿨 코튼 반팔 티셔츠"
              placeholderTextColor="#666"
              value={garmentDescription}
              onChangeText={setGarmentDescription}
            />

            <Text style={[styles.title, { marginTop: 16, fontSize: 20 }]}>핏 선택</Text>
            <View style={styles.fitRow}>
              {['slim', 'regular', 'overfit'].map((fit) => (
                <TouchableOpacity
                  key={fit}
                  style={[styles.fitButton, fitType === fit && styles.selectedFitButton]}
                  onPress={() => setFitType(fit)}
                >
                  <Text style={[styles.fitButtonText, fitType === fit && styles.selectedFitButtonText]}>
                    {fit === 'slim' ? '🧘 슬림핏' : fit === 'overfit' ? '🧥 오버핏' : '👕 정사이즈'}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <TouchableOpacity
              style={styles.upscaleRow}
              onPress={() => setUseUpscale(!useUpscale)}
            >
              <View style={[styles.checkbox, useUpscale && styles.checkboxOn]} />
              <Text style={styles.upscaleText}>🔍 고해상도 업스케일 (Real-ESRGAN, 느리지만 선명)</Text>
            </TouchableOpacity>

            <View style={styles.buttonRow}>
              <TouchableOpacity style={[styles.button, styles.secondaryButton]} onPress={() => setStep(STEPS.INPUT_URL)}>
                <Text style={styles.buttonText}>← 뒤로</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.button} onPress={processVirtualTryOn}>
                <Text style={styles.buttonText}>🤖 AI 합성 시작!</Text>
              </TouchableOpacity>
            </View>
          </ScrollView>
        )}

        {step === STEPS.AI_FITTING && (
          <View style={styles.centerContainer}>
            <View style={styles.fittingPreview}>
              <ExpoImage source={{ uri: userPhoto }} style={styles.previewSmall} contentFit="cover" cachePolicy="memory-disk" transition={200} />
              <Text style={styles.plusSign}>+</Text>
              <ExpoImage source={{ uri: selectedClothingImage }} style={styles.previewSmall} contentFit="cover" cachePolicy="memory-disk" transition={200} />
            </View>
            <ActivityIndicator size="large" color="#FF6B6B" style={{ marginTop: 24 }} />
            <Text style={styles.loadingTitle}>AI가 옷을 입히는 중...</Text>
            <Text style={styles.loadingSubtitle}>
              얼굴은 그대로 보존하고, 옷만 새로 합성합니다.{'\n'}
              (해상도 768×1024 · {useUpscale ? '업스케일 ON' : '기본'} · 2~4분 소요)
            </Text>
          </View>
        )}

        {step === STEPS.RESULT && (
          <View style={styles.resultContainer}>
            <Text style={styles.resultTitle}>✨ 피팅 완료!</Text>
            <View style={styles.resultImageWrap}>
              <ExpoImage
                source={{ uri: finalImage }}
                style={styles.resultImage}
                contentFit="contain"
                cachePolicy="memory-disk"
                transition={400}
                priority="high"
              />
            </View>
            <View style={styles.buttonRow}>
              <TouchableOpacity style={[styles.button, styles.secondaryButton]} onPress={resetAll}>
                <Text style={styles.buttonText}>🔄 다른 옷</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.button, styles.secondaryButton]} onPress={startOver}>
                <Text style={styles.buttonText}>📸 처음부터</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

// ─────────────────────────────────────────────────────────────
// Sub-components (memoized to prevent re-render storms)
// ─────────────────────────────────────────────────────────────
const BodyField = memo(function BodyField({ label, value, onChange, placeholder, optional }) {
  return (
    <View style={styles.bodyField}>
      <Text style={styles.bodyLabel}>{label}{optional ? '' : ' *'}</Text>
      <TextInput
        style={styles.bodyInput}
        value={value}
        onChangeText={onChange}
        placeholder={placeholder}
        placeholderTextColor="#555"
        keyboardType="numeric"
      />
    </View>
  );
});

const SizeTable = memo(function SizeTable({ sizeChart }) {
  if (!sizeChart || !sizeChart.sizes || !sizeChart.sizes.length) return null;
  const allKeys = new Set();
  sizeChart.sizes.forEach(s => Object.keys(s.measurements).forEach(k => allKeys.add(k)));
  const keys = Array.from(allKeys).slice(0, 6);
  return (
    <View>
      <View style={[styles.sizeRow, { backgroundColor: '#1A1A1A' }]}>
        <Text style={[styles.sizeCell, styles.sizeHeader]}>사이즈</Text>
        {keys.map(k => <Text key={k} style={[styles.sizeCell, styles.sizeHeader]}>{MEASURE_LABEL[k] || k}</Text>)}
      </View>
      {sizeChart.sizes.map((s, i) => (
        <View key={i} style={styles.sizeRow}>
          <Text style={[styles.sizeCell, styles.sizeLabel]}>{s.label}</Text>
          {keys.map(k => (
            <Text key={k} style={styles.sizeCell}>{s.measurements[k] ?? '-'}</Text>
          ))}
        </View>
      ))}
    </View>
  );
});

const MEASURE_LABEL = {
  shoulder: '어깨', chest: '가슴', sleeve: '소매', length: '총장',
  waist: '허리', hip: '힙', thigh: '허벅지', rise: '밑위', hem: '밑단',
};

// ─────────────────────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0D0D0D' },
  centerContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
  scrollCenter: { flexGrow: 1, alignItems: 'center', padding: 24, paddingTop: 40, paddingBottom: 60 },

  emoji: { fontSize: 64, marginBottom: 16 },
  title: { fontSize: 24, fontWeight: '700', color: '#FFFFFF', marginBottom: 8, textAlign: 'center' },
  subtitle: { fontSize: 14, color: '#999', marginBottom: 20, textAlign: 'center', lineHeight: 20 },
  tip: { fontSize: 12, color: '#666', marginTop: 12, textAlign: 'center' },

  button: { backgroundColor: '#4D96FF', paddingVertical: 14, paddingHorizontal: 24, borderRadius: 30 },
  primaryButton: { backgroundColor: '#4D96FF' },
  secondaryButton: { backgroundColor: '#2A2A2A', marginRight: 10 },
  buttonText: { color: '#FFFFFF', fontSize: 15, fontWeight: '600' },
  buttonRow: { flexDirection: 'row', marginTop: 20, flexWrap: 'wrap', justifyContent: 'center' },

  input: { width: '100%', backgroundColor: '#1A1A1A', color: '#FFF', padding: 16, borderRadius: 14, borderWidth: 1, borderColor: '#333', fontSize: 14, marginBottom: 12 },
  thumbnail: { width: 90, height: 90, borderRadius: 45, marginBottom: 20, borderWidth: 2, borderColor: '#4D96FF' },
  photoMeta: { fontSize: 11, color: '#888', marginBottom: 12, textAlign: 'center' },

  loadingTitle: { color: '#FFFFFF', marginTop: 20, fontSize: 18, fontWeight: '600', textAlign: 'center' },
  loadingSubtitle: { color: '#888', marginTop: 8, fontSize: 14, textAlign: 'center', lineHeight: 22 },

  fittingPreview: { flexDirection: 'row', alignItems: 'center' },
  previewSmall: { width: 100, height: 100, borderRadius: 12, borderWidth: 1, borderColor: '#333' },
  plusSign: { color: '#FF6B6B', fontSize: 32, fontWeight: '700', marginHorizontal: 16 },

  // Body input
  bodyGrid: { width: '100%', flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between' },
  bodyField: { width: '48%', marginBottom: 12 },
  bodyLabel: { color: '#BBB', fontSize: 13, marginBottom: 6, fontWeight: '600' },
  bodyInput: {
    backgroundColor: '#1A1A1A', color: '#FFF', padding: 12, borderRadius: 10,
    borderWidth: 1, borderColor: '#333', fontSize: 15,
  },

  // Size recommendation
  recommendCard: {
    width: '100%', backgroundColor: '#1A1A1A', borderRadius: 20,
    padding: 24, alignItems: 'center', marginTop: 16,
    borderWidth: 1, borderColor: '#4D96FF',
  },
  recommendLabel: { color: '#888', fontSize: 13, marginBottom: 4 },
  recommendSize: { color: '#4D96FF', fontSize: 64, fontWeight: '800', lineHeight: 70 },
  recommendFit:  { color: '#FFD93D', fontSize: 16, fontWeight: '700', marginTop: 4 },
  recommendDetail: { color: '#CCC', fontSize: 13, textAlign: 'center', marginTop: 8, lineHeight: 20 },
  measureRow: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', marginTop: 16 },
  measureChip: {
    backgroundColor: '#0D0D0D', paddingHorizontal: 10, paddingVertical: 6,
    borderRadius: 8, margin: 3,
  },
  measureKey: { color: '#888', fontSize: 11 },
  measureVal: { color: '#FFF', fontSize: 13, fontWeight: '600' },

  altRow: { flexDirection: 'row', width: '100%', marginTop: 12, justifyContent: 'space-between' },
  altCard: { flex: 1, backgroundColor: '#1A1A1A', padding: 16, borderRadius: 14, marginHorizontal: 4, alignItems: 'center' },
  altLabel: { color: '#888', fontSize: 12 },
  altSize:  { color: '#FFF', fontSize: 24, fontWeight: '700', marginTop: 4 },

  sizeTableBox: { width: '100%', backgroundColor: '#1A1A1A', borderRadius: 14, padding: 12, marginTop: 16 },
  sizeTableTitle: { color: '#FFF', fontSize: 14, fontWeight: '600', marginBottom: 8 },
  sizeRow: { flexDirection: 'row', paddingVertical: 6, borderBottomWidth: 1, borderBottomColor: '#222' },
  sizeCell: { flex: 1, color: '#CCC', fontSize: 12, textAlign: 'center' },
  sizeHeader: { color: '#888', fontWeight: '700' },
  sizeLabel: { color: '#4D96FF', fontWeight: '700' },

  // Carousel & fit
  carousel: { width: '100%', maxHeight: 150, marginVertical: 10 },
  carouselImage: { width: 100, height: 130, borderRadius: 8, marginRight: 12, borderWidth: 2, borderColor: 'transparent', opacity: 0.5 },
  selectedImage: { borderColor: '#FF6B6B', opacity: 1 },

  fitRow: { flexDirection: 'row', justifyContent: 'space-between', width: '100%', marginBottom: 12 },
  fitButton: { flex: 1, backgroundColor: '#1A1A1A', paddingVertical: 12, borderRadius: 10, alignItems: 'center', marginHorizontal: 4, borderWidth: 1, borderColor: '#333' },
  selectedFitButton: { backgroundColor: '#FF6B6B', borderColor: '#FF6B6B' },
  fitButtonText: { color: '#888', fontSize: 13, fontWeight: '600' },
  selectedFitButtonText: { color: '#FFF' },

  upscaleRow: { flexDirection: 'row', alignItems: 'center', marginTop: 12, alignSelf: 'flex-start' },
  checkbox: { width: 22, height: 22, borderRadius: 6, borderWidth: 2, borderColor: '#555', marginRight: 10 },
  checkboxOn: { backgroundColor: '#4D96FF', borderColor: '#4D96FF' },
  upscaleText: { color: '#CCC', fontSize: 13 },

  // Result
  resultContainer: { flex: 1, alignItems: 'center', padding: 16, paddingTop: 40 },
  resultTitle: { fontSize: 24, fontWeight: '700', color: '#FFFFFF', marginBottom: 12 },
  resultImageWrap: { flex: 1, width: '100%', backgroundColor: '#111', borderRadius: 16, overflow: 'hidden', marginBottom: 12 },
  resultImage: { width: '100%', height: '100%' },
});
