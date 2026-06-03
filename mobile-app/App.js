import React, { useState } from 'react';
import {
  StyleSheet, Text, View, TextInput, TouchableOpacity,
  Image, ActivityIndicator, Alert, SafeAreaView, ScrollView
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';

const BACKEND_URL = 'http://192.168.31.231:3000';

export default function App() {
  const [step, setStep] = useState('SELECT_PHOTO');
  // SELECT_PHOTO -> INPUT_URL -> SCRAPING -> IMAGE_SELECTION -> AI_FITTING -> RESULT
  
  const [userPhoto, setUserPhoto] = useState(null);
  const [userPhotoBase64, setUserPhotoBase64] = useState('');
  
  const [shoppingUrl, setShoppingUrl] = useState('');
  
  const [scrapedImages, setScrapedImages] = useState([]);
  const [selectedClothingImage, setSelectedClothingImage] = useState(null);
  const [garmentDescription, setGarmentDescription] = useState('');
  const [garmentType, setGarmentType] = useState('upper');
  const [fitType, setFitType] = useState('regular');
  const [features, setFeatures] = useState({});
  
  const [finalImage, setFinalImage] = useState(null);

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 0.5,
      base64: true,
    });

    if (!result.canceled) {
      setUserPhoto(result.assets[0].uri);
      setUserPhotoBase64(`data:image/jpeg;base64,${result.assets[0].base64}`);
      setStep('INPUT_URL');
    }
  };

  const scrapeClothing = async () => {
    if (!shoppingUrl) {
      Alert.alert('오류', 'URL을 입력해주세요.');
      return;
    }

    try {
      setStep('SCRAPING');
      const scrapeRes = await axios.post(`${BACKEND_URL}/api/scrape`,
        { url: shoppingUrl },
        { timeout: 15000, headers: { 'Bypass-Tunnel-Reminder': 'true' } }
      );

      const { imageUrls, garmentDescription: scrapedDesc, garmentType: scrapedType, features: scrapedFeatures } = scrapeRes.data;
      
      if (!imageUrls || imageUrls.length === 0) {
        throw new Error('옷 이미지를 찾을 수 없습니다.');
      }

      setScrapedImages(imageUrls);
      setSelectedClothingImage(imageUrls[0]);
      setGarmentDescription(scrapedDesc || '');
      setGarmentType(scrapedType || 'upper');
      setFeatures(scrapedFeatures || {});
      setFitType('regular');
      setStep('IMAGE_SELECTION');
      
    } catch (error) {
      const errMsg = error.response?.data?.error || error.message;
      Alert.alert('오류', '스크래핑 실패: ' + errMsg);
      setStep('INPUT_URL');
    }
  };

  const processVirtualTryOn = async () => {
    try {
      setStep('AI_FITTING');
      const tryonRes = await axios.post(`${BACKEND_URL}/api/tryon`, {
        userImageBase64: userPhotoBase64,
        clothingImageUrl: selectedClothingImage,
        garmentDescription: garmentDescription,
        garmentType: garmentType,
        fitType: fitType,
        features: features,
      }, {
        timeout: 300000, // 5분 (AI 합성 평균 2~3분 소요)
        headers: { 'Bypass-Tunnel-Reminder': 'true' },
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
      });

      if (tryonRes.data && tryonRes.data.resultImageUrl) {
        setFinalImage(tryonRes.data.resultImageUrl);
        setStep('RESULT');
      } else {
        throw new Error('AI 합성 결과를 받지 못했습니다.');
      }
    } catch (error) {
      const errMsg = error.response?.data?.error || error.message;
      if (error.code === 'ECONNABORTED') {
        Alert.alert('시간 초과', 'AI 서버가 바쁩니다. 잠시 후 다시 시도해주세요.');
      } else {
        Alert.alert('오류', errMsg);
      }
      setStep('IMAGE_SELECTION');
    }
  };

  const resetAll = () => {
    setShoppingUrl('');
    setScrapedImages([]);
    setSelectedClothingImage(null);
    setGarmentDescription('');
    setFinalImage(null);
    setStep('INPUT_URL');
  };

  const startOver = () => {
    setUserPhoto(null);
    setUserPhotoBase64('');
    resetAll();
    setStep('SELECT_PHOTO');
  };

  return (
    <SafeAreaView style={styles.container}>

      {step === 'SELECT_PHOTO' && (
        <View style={styles.centerContainer}>
          <Text style={styles.emoji}>👗</Text>
          <Text style={styles.title}>Virtual Fitting AI</Text>
          <Text style={styles.subtitle}>내 사진을 선택하면{'\n'}AI가 옷을 입혀드립니다</Text>
          <TouchableOpacity style={styles.button} onPress={pickImage}>
            <Text style={styles.buttonText}>📸 사진 선택</Text>
          </TouchableOpacity>
        </View>
      )}

      {step === 'INPUT_URL' && (
        <ScrollView contentContainerStyle={styles.scrollCenter} keyboardShouldPersistTaps="handled">
          <Image source={{ uri: userPhoto }} style={styles.thumbnail} />
          <Text style={styles.title}>어떤 옷을 입어볼까요?</Text>
          <Text style={styles.subtitle}>쇼핑몰 URL을 붙여넣으세요</Text>
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
            <TouchableOpacity style={[styles.button, styles.secondaryButton]} onPress={startOver}>
              <Text style={styles.buttonText}>← 뒤로</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={scrapeClothing}>
              <Text style={styles.buttonText}>옷 정보 가져오기 🔍</Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      )}

      {step === 'SCRAPING' && (
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color="#4D96FF" />
          <Text style={styles.loadingTitle}>정보를 수집하는 중...</Text>
          <Text style={styles.loadingSubtitle}>쇼핑몰에서 옷 사진과 상세 정보를 찾고 있습니다</Text>
        </View>
      )}

      {step === 'IMAGE_SELECTION' && (
        <ScrollView contentContainerStyle={styles.scrollCenter} keyboardShouldPersistTaps="handled">
          <Text style={styles.title}>합성할 옷 이미지를 선택하세요</Text>
          <Text style={styles.subtitle}>정면 사진이나 누끼컷을 고르면 퀄리티가 높아집니다.</Text>
          
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.carousel}>
            {scrapedImages.map((imgUrl, index) => (
              <TouchableOpacity key={index} onPress={() => setSelectedClothingImage(imgUrl)}>
                <Image 
                  source={{ uri: imgUrl }} 
                  style={[
                    styles.carouselImage, 
                    selectedClothingImage === imgUrl && styles.selectedImage
                  ]} 
                />
              </TouchableOpacity>
            ))}
          </ScrollView>

          <Text style={[styles.title, { marginTop: 24, fontSize: 20 }]}>옷 상세 설명 (AI 프롬프트)</Text>
          <Text style={styles.subtitle}>추출된 정보입니다. 핏이나 종류를 덧붙여 AI를 도울 수 있어요.</Text>
          <TextInput
            style={[styles.input, { height: 80, textAlignVertical: 'top' }]}
            multiline
            placeholder="예: 오버핏 쿨 코튼 반팔 티셔츠"
            placeholderTextColor="#666"
            value={garmentDescription}
            onChangeText={setGarmentDescription}
          />

          <Text style={[styles.title, { marginTop: 16, fontSize: 20 }]}>핏 선택 옵션</Text>
          <Text style={styles.subtitle}>원하는 피팅 스타일을 선택하세요</Text>
          <View style={styles.fitRow}>
            {['slim', 'regular', 'overfit'].map((fit) => (
              <TouchableOpacity
                key={fit}
                style={[
                  styles.fitButton,
                  fitType === fit && styles.selectedFitButton
                ]}
                onPress={() => setFitType(fit)}
              >
                <Text style={[
                  styles.fitButtonText,
                  fitType === fit && styles.selectedFitButtonText
                ]}>
                  {fit === 'slim' ? '🧘 슬림핏' : fit === 'overfit' ? '🧥 오버핏' : '👕 정사이즈'}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          <View style={styles.buttonRow}>
            <TouchableOpacity style={[styles.button, styles.secondaryButton]} onPress={() => setStep('INPUT_URL')}>
              <Text style={styles.buttonText}>← 뒤로</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={processVirtualTryOn}>
              <Text style={styles.buttonText}>🤖 AI 합성 시작!</Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      )}

      {step === 'AI_FITTING' && (
        <View style={styles.centerContainer}>
          <View style={styles.fittingPreview}>
            <Image source={{ uri: userPhoto }} style={styles.previewSmall} />
            <Text style={styles.plusSign}>+</Text>
            <Image source={{ uri: selectedClothingImage }} style={styles.previewSmall} />
          </View>
          <ActivityIndicator size="large" color="#FF6B6B" style={{ marginTop: 24 }} />
          <Text style={styles.loadingTitle}>AI가 프롬프트를 분석 중...</Text>
          <Text style={styles.loadingSubtitle}>
            "{garmentDescription.substring(0, 30)}{garmentDescription.length > 30 ? '...' : ''}"{'\n'}
            정보를 바탕으로 완벽한 핏을 만들고 있습니다. (2~3분 소요)
          </Text>
        </View>
      )}

      {step === 'RESULT' && (
        <View style={styles.resultContainer}>
          <Text style={styles.resultTitle}>✨ 피팅 완료!</Text>
          <View style={styles.resultImageWrap}>
            <Image source={{ uri: finalImage }} style={styles.resultImage} resizeMode="contain" />
          </View>
          <View style={styles.buttonRow}>
            <TouchableOpacity style={[styles.button, styles.secondaryButton]} onPress={resetAll}>
              <Text style={styles.buttonText}>🔄 다른 옷</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.button, styles.secondaryButton]} onPress={startOver}>
              <Text style={styles.buttonText}>📸 다른 사진</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={() => Alert.alert('저장', '기능 준비중입니다.')}>
              <Text style={styles.buttonText}>💾 저장</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0D0D0D' },
  centerContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
  scrollCenter: { flexGrow: 1, justifyContent: 'center', alignItems: 'center', padding: 24, paddingTop: 40 },
  emoji: { fontSize: 64, marginBottom: 16 },
  title: { fontSize: 24, fontWeight: '700', color: '#FFFFFF', marginBottom: 8, textAlign: 'center' },
  subtitle: { fontSize: 14, color: '#999', marginBottom: 20, textAlign: 'center', lineHeight: 20 },
  button: { backgroundColor: '#4D96FF', paddingVertical: 14, paddingHorizontal: 24, borderRadius: 30 },
  secondaryButton: { backgroundColor: '#2A2A2A', marginRight: 10 },
  buttonText: { color: '#FFFFFF', fontSize: 15, fontWeight: '600' },
  buttonRow: { flexDirection: 'row', marginTop: 20, flexWrap: 'wrap', justifyContent: 'center' },
  input: { width: '100%', backgroundColor: '#1A1A1A', color: '#FFF', padding: 16, borderRadius: 14, borderWidth: 1, borderColor: '#333', fontSize: 14, marginBottom: 12 },
  thumbnail: { width: 90, height: 90, borderRadius: 45, marginBottom: 20, borderWidth: 2, borderColor: '#4D96FF' },
  loadingTitle: { color: '#FFFFFF', marginTop: 20, fontSize: 18, fontWeight: '600', textAlign: 'center' },
  loadingSubtitle: { color: '#888', marginTop: 8, fontSize: 14, textAlign: 'center', lineHeight: 22 },
  fittingPreview: { flexDirection: 'row', alignItems: 'center' },
  previewSmall: { width: 100, height: 100, borderRadius: 12, borderWidth: 1, borderColor: '#333' },
  plusSign: { color: '#FF6B6B', fontSize: 32, fontWeight: '700', marginHorizontal: 16 },
  resultContainer: { flex: 1, alignItems: 'center', padding: 16, paddingTop: 40 },
  resultTitle: { fontSize: 24, fontWeight: '700', color: '#FFFFFF', marginBottom: 12 },
  resultImageWrap: { flex: 1, width: '100%', backgroundColor: '#111', borderRadius: 16, overflow: 'hidden', marginBottom: 12 },
  resultImage: { width: '100%', height: '100%' },
  carousel: { width: '100%', maxHeight: 150, marginVertical: 10 },
  carouselImage: { width: 100, height: 130, borderRadius: 8, marginRight: 12, borderWidth: 2, borderColor: 'transparent', opacity: 0.5 },
  selectedImage: { borderColor: '#FF6B6B', opacity: 1 },
  fitRow: { flexDirection: 'row', justifyContent: 'space-between', width: '100%', marginBottom: 12 },
  fitButton: { flex: 1, backgroundColor: '#1A1A1A', paddingVertical: 12, borderRadius: 10, alignItems: 'center', marginHorizontal: 4, borderWidth: 1, borderColor: '#333' },
  selectedFitButton: { backgroundColor: '#FF6B6B', borderColor: '#FF6B6B' },
  fitButtonText: { color: '#888', fontSize: 13, fontWeight: '600' },
  selectedFitButtonText: { color: '#FFF' },
});
