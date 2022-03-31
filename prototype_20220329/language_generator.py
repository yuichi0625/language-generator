"""人工言語生成器を実装する"""
import itertools
from math import inf, log

from scipy.stats import norm

from datatypes import CharType, Sentence
from ngram import NgramCharTypeModels

# math.log(0)を回避するため
E = 1e-7
# 文字の位置の確率分布がスコアに影響する割合
# 確率分布自体はscipy.stats.norm(loc=0, scale=NORM_SCALE)で計算されている
DIST_POS_RATIO = 1.0
# 大きくしすぎると現在位置から遠い部分の確率が上がるので、文が短くなりがち
NORM_SCALE = 2.0
# CharType.SYMBOLの出力確率の補正倍率
# スペースが出力されにくいので用意した
# 値を大きくしすぎるとSYMBOLを取りにいくため文が短くなりがち
SYMBOL_MAGNIFICATION = 3.0
# 次のCharTypeの確率分布がスコアに影響する割合
DIST_CHARTYPE_RATIO = 1.0
# 出力済み文字数から算出した確率分布がスコアに影響する割合
# できるだけ平等に全Sentenceから取得してこようとする力が働く
# そのため全体的に文字を取得できるが、文が短くなりがち
DIST_USED_RATIO = 0.1
# 未出力の文字数から算出した確率分布がスコアに影響する割合
# より長いSentenceから取得してこようとする力が働く
# そのため最初の方に集中した長いSentenceから文字を取得しがち
DIST_LEFT_RATIO = 1.0


class LanguageGenerator:
    """人工言語生成器

    generator = LanguageGenerator()
    generator.fit(sentences)
    new_sentence = generator.generate()
    """

    def __init__(self) -> None:
        self.sentences = None
        self.curr_ids = None
        self.num_used = None
        self.num_left = None
        self.ngram_models = None
        self.thresh_symbol = None
        self.dist_pos = None

    def fit(self, sentences: list[str]) -> None:
        """入力文"""
        self.sentences = [
            Sentence(sentence, add_bos=2, add_eos=1) for sentence in sentences
        ]
        # 各Sentenceの現在位置を記録する
        self.curr_ids = [1 for _ in range(len(self.sentences))]
        # 各Sentenceから出力した文字数を記録する
        self.num_used = [0 for _ in range(len(self.sentences))]
        # 各Sentenceで未出力の文字数を記録する
        self.num_left = [
            sentence.get_num_chars() for sentence in self.sentences
        ]

        # CharTypeの確率分布を作成する
        self.ngram_models = NgramCharTypeModels(ngrams=[2, 3])
        self.ngram_models.fit(self.sentences)
        # CharType.Symbolが出力されにくいため、一定間隔出力されていないければ出現確率を上げたい
        # そのための出力間隔の閾値を求める
        self.thresh_symbol = self._calc_symbol_threshold()
        # 現在位置を基準に周囲の文字が出力される確率分布を作成する
        max_length = max([len(sent) for sent in self.sentences])
        self.dist_pos = norm.pdf(range(max_length), loc=0, scale=NORM_SCALE)

    def generate(self) -> str:
        """fitで入力した文から人工言語を生成する"""
        output = Sentence('', add_bos=2)

        while output[-1].chartype != CharType.EOS:
            scores = self._init_scores()
            self._update_scores(scores, output)
            sent_id, char_id = self._extract_next_char_index(scores)
            output.append(self.sentences[sent_id][char_id])

            self.num_used[sent_id] += 1
            self.num_left[sent_id] -= 1
            # 各Sentenceの長さに合わせてcurr_idsを更新する
            ratio = (char_id + 1) / len(self.sentences[sent_id])
            self.curr_ids = [
                round(len(sentence) * ratio) - 1 for sentence in self.sentences
            ]

        return ''.join([char.char for char in output.get_chars()])

    def _calc_symbol_threshold(self) -> float:
        """CharType.Symbolの出力確率を上げる、出力間隔の閾値を求める"""
        num_chars = sum(
            [sentence.get_num_chars() for sentence in self.sentences])
        num_symbols = len([
            char for char in itertools.chain.from_iterable(self.sentences)
            if char.chartype == CharType.SYMBOL
        ])
        thresh_symbol = num_chars / num_symbols

        return thresh_symbol

    def _init_scores(self) -> list[list[float]]:
        """対数スコアを初期化する

        curr_ids以下の部分については出力されないよう-infにする
        """
        lengths = [len(sentence) for sentence in self.sentences]

        scores = []
        for length, curr_id in zip(lengths, self.curr_ids):
            score = [-inf if i <= curr_id else 0.0 for i in range(length)]
            scores.append(score)

        return scores

    def _update_scores(self, scores: list[list[float]],
                       output: Sentence) -> None:
        """対数スコアを更新する"""
        # CharTypeの確率分布を求める
        dist_chartype = self._get_chartype_dist(output)
        # 出力した文字数をもとに各Sentenceから文字を出力する確率分布を計算する
        inv_num_used = [max(self.num_used) + 1 - num for num in self.num_used]
        dist_used = [num / sum(inv_num_used) for num in inv_num_used]
        # 未出力の文字数をもとに各Sentenceから文字を出力する確率分布を計算する
        dist_left = [num / sum(self.num_left) for num in self.num_left]

        for sent_id, sentence in enumerate(self.sentences):
            curr_id = self.curr_ids[sent_id]
            for i, char in enumerate(sentence[curr_id:]):
                score = 0
                score += log(self.dist_pos[i] + E) * DIST_POS_RATIO
                score += log(dist_chartype[char.chartype] +
                             E) * DIST_CHARTYPE_RATIO
                score += log(dist_used[sent_id] + E) * DIST_USED_RATIO
                score += log(dist_left[sent_id] + E) * DIST_LEFT_RATIO
                scores[sent_id][i + curr_id] += score

    def _get_chartype_dist(self, output: Sentence) -> dict[CharType, float]:
        """CharTypeの確率分布を求める"""
        # CharType.Symbolの確率分布の倍率を求める
        mag_symbol = self._calc_symbol_magnification(output)
        # CharTypeの確率分布を求める
        dist_chartype = self.ngram_models.get_dist(output[-2:])
        dist_chartype[CharType.SYMBOL] *= mag_symbol

        return dist_chartype

    def _calc_symbol_magnification(self, output: Sentence) -> float:
        """CharType.Symbolの確率分布の倍率を求める"""
        length = 0
        for char in output[::-1]:
            if char.chartype in {CharType.SYMBOL, CharType.EOS}:
                break
            length += 1

        return SYMBOL_MAGNIFICATION if length > self.thresh_symbol else 1.0

    def _extract_next_char_index(self,
                                 scores: list[list[float]]) -> tuple[int, int]:
        """次に出力される文字のindexを取得する"""
        sent_id = 0
        char_id = 0
        max_score = -inf
        for i in range(len(self.sentences)):
            for j, score in enumerate(scores[i]):
                if score > max_score:
                    sent_id = i
                    char_id = j
                    max_score = score

        return sent_id, char_id
