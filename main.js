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

const WINDOW_SIZE = 2;
const EMBEDDING_SIZE = 10;
const LEARNING_RATE = 0.01;
const EPOCHS = 1000;

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

function randomMatrix(rows, cols) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => Math.random()),
  );
}

function softmax(logits) {
  // 수치 안정성 개선
  const maxLogit = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - maxLogit));
  const sumExp = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / sumExp);
}

function forward(inputIndex, W1, W2, vocabSize, embeddingSize) {
  const hidden = W1[inputIndex];
  const scores = new Array(vocabSize).fill(0);

  for (let vocabIndex = 0; vocabIndex < vocabSize; vocabIndex += 1) {
    let score = 0;

    for (let dim = 0; dim < embeddingSize; dim += 1) {
      score += hidden[dim] * W2[dim][vocabIndex];
    }

    scores[vocabIndex] = score;
  }

  return { hidden, scores };
}

function getLoss(predictedProbabilities, targetIndex) {
  return -Math.log(predictedProbabilities[targetIndex] + 1e-12);
}

function backward(inputIndex, targetIndex, hidden, probs, W1, W2, learningRate) {
  // 1) dScores 계산
  const dScores = [...probs];
  dScores[targetIndex] -= 1;

  // 2) dHidden 계산 (W2 업데이트 전에 계산)
  const dHidden = new Array(hidden.length).fill(0);
  for (let dim = 0; dim < hidden.length; dim += 1) {
    for (let vocabIndex = 0; vocabIndex < dScores.length; vocabIndex += 1) {
      dHidden[dim] += W2[dim][vocabIndex] * dScores[vocabIndex];
    }
  }

  // 3) W2 업데이트
  for (let dim = 0; dim < hidden.length; dim += 1) {
    for (let vocabIndex = 0; vocabIndex < dScores.length; vocabIndex += 1) {
      W2[dim][vocabIndex] -= learningRate * hidden[dim] * dScores[vocabIndex];
    }
  }

  // 4) W1 업데이트 (input row only)
  for (let dim = 0; dim < hidden.length; dim += 1) {
    W1[inputIndex][dim] -= learningRate * dHidden[dim];
  }
}

function train(trainingData, epochs, learningRate, W1, W2, vocabSize, embeddingSize) {
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    let totalLoss = 0;

    for (const { input, target } of trainingData) {
      const { hidden, scores } = forward(input, W1, W2, vocabSize, embeddingSize);
      const probs = softmax(scores);
      totalLoss += getLoss(probs, target);
      backward(input, target, hidden, probs, W1, W2, learningRate);
    }

    if (epoch % 100 === 0) {
      const avgLoss = totalLoss / trainingData.length;
      console.log(`epoch ${epoch} loss: ${avgLoss.toFixed(4)}`);
    }
  }
}

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
}

main();