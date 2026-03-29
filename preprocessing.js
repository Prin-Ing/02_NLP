/** 문장 문자열 배열 -> 토큰 배열 */
function tokenize(sentences) {
  return sentences.map((sentence) => sentence.trim().split(/\s+/));
}

/**
 * 단어 사전(word -> index) 생성
 * - 코퍼스에서 처음 등장한 순서대로 인덱스 부여
 */
function buildWordToIndex(tokenizedText) {
  const wordToIndex = {};
  let currentIndex = 0;

  for (const sentence of tokenizedText) {
    for (const word of sentence) {
      if (wordToIndex[word] !== undefined) continue;
      wordToIndex[word] = currentIndex;
      currentIndex += 1;
    }
  }

  return wordToIndex;
}

/** 토큰 배열 -> 정수 인덱스 배열 */
function encode(tokenizedText, wordToIndex) {
  return tokenizedText.map((sentence) => sentence.map((word) => wordToIndex[word]));
}

/** 정수 인덱스 배열 -> 토큰 배열 */
function decode(indices, wordToIndex) {
  const indexToWord = Object.entries(wordToIndex)
    .sort(([, indexA], [, indexB]) => indexA - indexB)
    .map(([word]) => word);

  return indices.map((index) => indexToWord[index]);
}

/** [이론용] 원-핫 인코딩 */
function oneHotEncode(encodedText, vocabSize) {
  return encodedText.map((sentence) =>
    sentence.map((wordIndex) => {
      const oneHotVector = new Array(vocabSize).fill(0);
      oneHotVector[wordIndex] = 1;
      return oneHotVector;
    }),
  );
}

/** CBOW/Skip-gram용 (input, target) 쌍 생성 */
function makeTrainingData(encodedText, windowSize) {
  const trainingData = [];

  for (let sentenceIndex = 0; sentenceIndex < encodedText.length; sentenceIndex += 1) {
    const sentence = encodedText[sentenceIndex];

    for (let centerIndex = 0; centerIndex < sentence.length; centerIndex += 1) {
      const input = sentence[centerIndex];

      for (let offset = -windowSize; offset <= windowSize; offset += 1) {
        if (offset === 0) continue;

        const targetIndex = centerIndex + offset;
        if (targetIndex < 0 || targetIndex >= sentence.length) continue;

        const target = sentence[targetIndex];
        trainingData.push({ input, target });
      }
    }
  }

  return trainingData;
}

module.exports = {
  tokenize,
  buildWordToIndex,
  encode,
  decode,
  oneHotEncode,
  makeTrainingData,
};
