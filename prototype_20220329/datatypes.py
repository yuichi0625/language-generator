"""データ型を実装する"""
from __future__ import annotations

from dataclasses import dataclass, field, InitVar
from enum import auto, Enum
from pathlib import Path
from typing import Optional, Union

from unidecode import unidecode

BOS = '<BOS>'
EOS = '<EOS>'

VOWELS_PATH = 'data/vowels.txt'
VOWELS = set(Path(VOWELS_PATH).read_text(encoding='utf-8').splitlines())


class CharType(Enum):
    """母音・子音を軸とした文字の種類"""
    BOS = auto()  # Begin Of Sentence
    EOS = auto()  # End Of Sentence
    CONSONANT = auto()
    VOWEL = auto()
    SYMBOL = auto()  # スペースや句読点を想定


@dataclass
class Sentence:
    """文章を表現するデータ型"""
    chars: list[Char] = field(init=False)
    sentence: InitVar[str]
    add_bos: InitVar[Optional[int]] = None
    add_eos: InitVar[Optional[int]] = None

    def __post_init__(self, sentence: str, add_bos: Optional[int],
                      add_eos: Optional[int]):
        self.chars = []
        if add_bos is not None:
            self.extend([BOS] * add_bos)
        self.extend(list(sentence))
        if add_eos is not None:
            self.extend([EOS] * add_eos)

    def __getitem__(self, key):
        return self.chars[key]

    def __iter__(self):
        for char in self.chars:
            yield char

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return ''.join([char.char for char in self.chars])

    def append(self, char: Union[str, Char]) -> None:
        """文字を追加する"""
        char_ = Char(char) if isinstance(char, str) else char
        self.chars.append(char_)

    def extend(self, chars: list[Union[str, Char]]) -> None:
        """複数の文字を追加する"""
        for char in chars:
            self.append(char)

    def get_chars(self) -> list[Char]:
        """BOS,EOS以外の文字を取得する"""
        return [
            char for char in self.chars
            if char.chartype not in {CharType.BOS, CharType.EOS}
        ]

    def get_num_chars(self) -> int:
        """BOS,EOS以外の文字数を取得する"""
        return len(self.get_chars())


@dataclass
class Char:
    """文字を表現するデータ型"""
    char: str
    chartype: CharType = field(init=False)

    def __post_init__(self):
        # unidecode.unidecodeを使うと、アクセント記号を削除した上でラテンアルファベットに変換してくれる
        ascii_ = unidecode(self.char)
        if ascii_ == BOS:
            self.chartype = CharType.BOS
        elif ascii_ == EOS:
            self.chartype = CharType.EOS
        elif not ascii_.isalpha():
            self.chartype = CharType.SYMBOL
        # 変換後の最後のアルファベットが母音か子音かで判定を行う
        elif ascii_[-1].lower() in VOWELS:
            self.chartype = CharType.VOWEL
        else:
            self.chartype = CharType.CONSONANT

    def __str__(self):
        return self.char
