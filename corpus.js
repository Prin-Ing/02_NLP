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

module.exports = {
  corpus,
  WINDOW_SIZE,
  EMBEDDING_SIZE,
  LEARNING_RATE,
  EPOCHS,
};
