// 간단한 한국어 코퍼스(문장 집합)
// - 단어 임베딩/인코딩 전처리 과정을 연습하기 위한 예시 데이터
const corpus = [
  '왕 강하다 용감하다',
  '여왕 아름답다 지혜롭다',
  '왕자 용감하다 젊다',
  '공주 아름답다 친절하다',
  '왕 여왕 궁전 살다',
  '왕자 공주 젊다 아름답다',
  '개 충성스럽다 귀엽다',
  '고양이 독립적이다 귀엽다',
  '강아지 귀엽다 활발하다',
  '개 고양이 동물',
  '강아지 개 동물',
  '왕 남자 지도자',
  '여왕 여자 지도자',
  '왕자 남자 젊다',
  '공주 여자 젊다',
];

/**
 * 문장 단위 문자열 코퍼스를 토큰(단어) 배열로 분리
 * ex) "왕 강하다 용감하다" -> ["왕", "강하다", "용감하다"]
 */
function tokenize(corpus) {
  let text = [];

  for (let i = 0; i < corpus.length; i++) {
    text.push(corpus[i].split(' '));
  }

  return text;
}

/**
 * 단어 사전(word -> index) 생성
 * - 코퍼스에서 처음 등장한 순서대로 인덱스 부여
 * ex) { 왕: 0, 강하다: 1, ... }
 */
function word2idx(tokenizedText) {
  let indexedText = {};
  let index = 0;

  for (const sentence of tokenizedText) {
    for (const word of sentence) {
      if (word in indexedText) continue; // 이미 사전에 있으면 건너뜀
      indexedText[word] = index;
      index++;
    }
  }

  return indexedText;
}

/**
 * 토큰(단어) 배열을 정수 인덱스 배열로 변환
 * ex) ["왕", "강하다"] -> [0, 1]
 */
function encode(tokenizedText, indexedText) {
  let encodedText = [];

  for (const sentence of tokenizedText) {
    let encodedSentence = [];
    for (const word of sentence) {
      encodedSentence.push(indexedText[word]);
    }
    encodedText.push(encodedSentence);
  }

  return encodedText;
}

/**
 * [이론용] 원-핫 인코딩
 * - 각 단어 인덱스를 vocabSize 길이의 벡터로 변환
 * - 해당 인덱스 위치만 1, 나머지는 0
 *
 * 주의:
 * - 실무/학습에서는 메모리 비효율이 커서 보통 직접 원-핫을 만들지 않음
 * - 임베딩 레이어(Embedding)나 희소 표현을 주로 사용
 */
function oneHotEncode(encodedText, vocabSize) {
  let oneHotEncodedText = [];

  for (const sentence of encodedText) {
    let oneHotEncodedSentence = [];

    for (const wordIndex of sentence) {
      let oneHotVector = new Array(vocabSize).fill(0);
      oneHotVector[wordIndex] = 1;
      oneHotEncodedSentence.push(oneHotVector);
    }

    oneHotEncodedText.push(oneHotEncodedSentence);
  }

  return oneHotEncodedText;
}

// 1) 토큰화
const tokenizedText = tokenize(corpus);

// 2) 단어 사전 생성
const indexedText = word2idx(tokenizedText);

// 3) 정수 인코딩
const encodedText = encode(tokenizedText, indexedText);

// 확인 출력
console.log(encodedText);