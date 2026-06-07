# Expo HAS CHANGED

Read the exact versioned docs at https://docs.expo.dev/versions/v56.0.0/ before writing any code.

## @tensorflow/tfjs-react-native 트러블슈팅

"비전 모델을 안 받았다" / "model not received" 오류가 발생하면 아래 순서로 점검:

1. **Expo Go에서 실행 중인가?**
   tfjs-react-native는 네이티브 모듈이 필요하므로 Expo Go에서 작동 안 함.
   `npx expo install expo-dev-client` 후 `npx expo run:android` 또는 `npx expo run:ios`로 dev client 빌드 필요.

2. **모델 파일이 번들에 포함돼 있는가?**
   `assets/` 폴더에 `.bin`/`.tflite`/`.json`을 두고 Metro `resolver.assetExts`에 확장자 추가:
   ```js
   // metro.config.js
   const { getDefaultConfig } = require('expo/metro-config');
   const config = getDefaultConfig(__dirname);
   config.resolver.assetExts.push('bin', 'tflite');
   module.exports = config;
   ```
   `require('../assets/model/model.json')` 형태로 import.

3. **원격 모델 URL 사용 시 fetch 실패 가능**
   - HTTPS만 사용 (HTTP는 ATS에서 차단)
   - CORS / 인증 헤더 점검
   - `await tf.ready()` 호출 후 `tf.loadGraphModel(url)` await 확인

4. **React Native 0.85 + tfjs-react-native 1.0.0 호환성**
   SDK 56 (RN 0.85) 환경에서는 tfjs-react-native 1.0.0이 New Architecture (Fabric/TurboModules) 미대응일 수 있음.
   `app.json`에서 `"newArchEnabled": false`로 임시 회피하거나, 업그레이드된 fork 사용.

5. **이 프로젝트에서의 대안**
   - 클라이언트 측 사진 전처리: `src/vision/imagePreprocess.js` (expo-image-manipulator 기반, TFLite 불필요)
   - 비전 모델 추론은 모두 Python 백엔드에서 실행 (MediaPipe, SegFormer, IP-Adapter)
