"""n-gram言語モデルを実装する"""
import itertools

from datatypes import Char, CharType, Sentence


class NgramCharTypeModels:
    """NgramCharTypeModelを複数使用する際のラッパー

    確率分布は複数のNgramCharTypeModelの平均を出力する
    """

    def __init__(self, ngrams):
        if isinstance(ngrams, str):
            ngrams = [ngrams]

        self.ngrams = ngrams
        self.ngram_models = [NgramCharTypeModel(ngram) for ngram in ngrams]

    def fit(self, sentences):
        """入力文から言語モデルを作成する"""
        for ngram_model in self.ngram_models:
            ngram_model.fit(sentences)

    def get_dist(self, chars):
        """文字の出現確率を取得する"""
        if isinstance(chars, Char):
            chars = [chars]

        assert len(chars) == max(self.ngrams) - 1

        dists = [
            ngram_model.get_dist(chars[1 - ngram:])
            for ngram, ngram_model in zip(self.ngrams, self.ngram_models)
        ]
        dist = {
            chartype: sum([dist[chartype] for dist in dists]) / len(dists)
            for chartype in CharType
        }

        return dist


class NgramCharTypeModel:
    """CharTypeに関するn-gram言語モデル"""

    def __init__(self, ngram):
        self.dist = {}
        self.ngram = ngram

    def fit(self, sentences_):
        """入力文から言語モデルを作成する"""
        sentences = [
            Sentence(sentence, add_bos=self.ngram - 1, add_eos=1)
            for sentence in sentences_
        ]

        # カウントベースの言語モデルを作成する
        count_dict = self._init_model()
        self._count_chartypes(count_dict, sentences)

        # カウントベースを確率に変換する
        self.dist = self._convert_count_to_prob(count_dict)

    def get_dist(self, chars):
        """文字の出現確率を取得する"""
        if isinstance(chars, Char):
            chars = [chars]

        assert len(chars) == self.ngram - 1

        return recursive_get(self.dist, [char.chartype for char in chars])

    def _init_model(self):
        """言語モデルを初期化する"""
        count_dict = {}
        for prod in itertools.product(list(CharType), repeat=self.ngram):
            recursive_set(count_dict, prod, 1)

        return count_dict

    def _count_chartypes(self, count_dict, sentences):
        """入力文のCharTypeをカウントする"""
        for sentence in sentences:
            chartypes = [char.chartype for char in sentence]
            for i in range(len(chartypes) - self.ngram + 1):
                recursive_add(count_dict, chartypes[i:i + self.ngram], 1)

    def _convert_count_to_prob(self, count_dict):
        """カウントを確率に変換する"""
        dist = {}
        for prod in itertools.product(list(CharType), repeat=self.ngram - 1):
            chartype2count = recursive_get(count_dict, prod)
            total = sum(chartype2count.values())
            chartype2prob = {
                chartype: count / total
                for chartype, count in chartype2count.items()
            }
            recursive_set(dist, prod, chartype2prob)

        return dist


def recursive_set(dict_, keys, value):
    """keys分だけ下って値を代入する、すでに値がある場合は変更しない"""
    if not keys:
        return None

    key, *keys = keys
    if dict_.get(key) is None:
        dict_[key] = {} if keys else value
    if isinstance(dict_[key], dict):
        return recursive_set(dict_[key], keys, value)


def recursive_add(dict_, keys, value):
    """keys分だけ下って値を足す"""
    key, *keys = keys
    if not keys:
        dict_[key] += value
    else:
        return recursive_add(dict_[key], keys, value)


def recursive_get(dict_, keys):
    """keys分だけ下って値を取得する"""
    key, *keys = keys
    if not keys:
        return dict_[key]
    return recursive_get(dict_[key], keys)
