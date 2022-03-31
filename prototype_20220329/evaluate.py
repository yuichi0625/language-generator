"""出力結果を確認する"""
import argparse

import pandas as pd

from language_generator import (LanguageGenerator, DIST_POS_RATIO, NORM_SCALE,
                                SYMBOL_MAGNIFICATION, DIST_CHARTYPE_RATIO,
                                DIST_USED_RATIO, DIST_LEFT_RATIO)


def main() -> None:
    """main関数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path',
                        '-o',
                        default='output.txt',
                        type=str,
                        help='出力先')
    args = parser.parse_args()

    dataframe = pd.read_excel('data/eval_data.xlsx', header=0)
    sentences_list = [
        # 英語,フランス語,スペイン語,ポルトガル語
        dataframe.iloc[0, 0:4].tolist(),
        dataframe.iloc[1, 0:4].tolist(),
        dataframe.iloc[2, 0:4].tolist(),
        dataframe.iloc[3, 0:4].tolist(),
        # 上記＋ドイツ語
        dataframe.iloc[0, 0:5].tolist(),
        dataframe.iloc[1, 0:5].tolist(),
        dataframe.iloc[2, 0:5].tolist(),
        dataframe.iloc[3, 0:5].tolist(),
        # 上記＋ロシア語,ウクライナ語,ブルガリア語,モンゴル語
        dataframe.iloc[0, 0:9].tolist(),
        dataframe.iloc[1, 0:9].tolist(),
        dataframe.iloc[2, 0:9].tolist(),
        dataframe.iloc[3, 0:9].tolist(),
        # 韓国語,タイ語,ギリシア語,アムハラ語
        dataframe.iloc[0, 9:13].tolist(),
        dataframe.iloc[1, 9:13].tolist(),
        dataframe.iloc[2, 9:13].tolist(),
        dataframe.iloc[3, 9:13].tolist(),
        # 英語,フランス語,スペイン語,ポルトガル語,ドイツ語で長い文章
        [
            "Learn from yesterday, live for today, hope for tomorrow. The important thing is not to stop questioning.",
            "Apprendre d'hier, vivre pour aujourd'hui, espérer pour demain. L'important est de ne pas cesser de se remettre en question.",
            "Aprende del ayer, vive para el hoy, espera para el mañana. Lo importante es no dejar de cuestionar.",
            "Aprenda com o ontem, viva o hoje, espere pelo amanhã. O importante é não parar de questionar.",
            "Lerne von gestern, lebe für heute, hoffe auf morgen. Wichtig ist, nicht aufzuhören zu hinterfragen.",
        ]
    ]

    with open(args.output_path, 'w', encoding='utf-8') as out:
        out.write(f'DIST_POS_RATIO = {DIST_POS_RATIO}\n')
        out.write(f'NORM_SCALE = {NORM_SCALE}\n')
        out.write(f'SYMBOL_MAGNIFICATION = {SYMBOL_MAGNIFICATION}\n')
        out.write(f'DIST_CHARTYPE_RATIO = {DIST_CHARTYPE_RATIO}\n')
        out.write(f'DIST_USED_RATIO = {DIST_USED_RATIO}\n')
        out.write(f'DIST_LEFT_RATIO = {DIST_LEFT_RATIO}\n')
        out.write(f'\n{"=" * 30}\n\n')

        for sentences in sentences_list:
            for sentence in sentences:
                out.write(f'{len(sentence)} {sentence}\n')

            generator = LanguageGenerator()
            generator.fit(sentences)
            output = generator.generate()

            out.write(f'{"+" * 20}\n')
            out.write(f'{len(output)} {output}\n')
            out.write(f'\n{"=" * 30}\n\n')


if __name__ == '__main__':
    main()
