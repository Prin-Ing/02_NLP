const {
  corpus,
  WINDOW_SIZE,
  EMBEDDING_SIZE,
  LEARNING_RATE,
  EPOCHS,
} = require('./corpus');
const {
  tokenize,
  buildWordToIndex,
  encode,
  decode,
  oneHotEncode,
  makeTrainingData,
} = require('./preprocessing');
const { randomMatrix, train } = require('./model');

function main() {
  // 1) 토큰화
  const tokenizedText = tokenize(corpus);

  // 2) 단어 사전 생성
  const wordToIndex = buildWordToIndex(tokenizedText);

  // 3) 정수 인코딩
  const encodedText = encode(tokenizedText, wordToIndex);

  const vocabSize = Object.keys(wordToIndex).length;
  const W1 = randomMatrix(vocabSize, EMBEDDING_SIZE);
  const W2 = randomMatrix(EMBEDDING_SIZE, vocabSize);

  const trainingData = makeTrainingData(encodedText, WINDOW_SIZE);
  train(trainingData, EPOCHS, LEARNING_RATE, W1, W2, vocabSize, EMBEDDING_SIZE);

  // 샘플 디코드 확인
  console.log('decode sample:', decode(encodedText[0], wordToIndex));

  // 필요 시 원-핫 확인
  // console.log(oneHotEncode(encodedText, vocabSize));
  void oneHotEncode;
}

main();
