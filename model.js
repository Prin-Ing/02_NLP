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

module.exports = {
  randomMatrix,
  train,
};
