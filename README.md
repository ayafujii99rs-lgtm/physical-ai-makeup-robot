# 化粧ロボット 3D版

## 始め方
Makeup_Robot_3D_Colab.ipynb を https://colab.research.google.com/ にアップロードし、上から実行してください。旧ノートブックとは別に開きます。コードは埋め込み済みです。画像の選択時には、このリポジトリの portraits.png をアップロードしてください。GPUは不要。Python 3.11の専用環境を自動作成します。Colab本体のPythonは変更しません。

まず4秒の確認動画を生成し、次に本番80秒（4人×2方式×2頬×5秒）を生成します。1頬ごとにログが出ます。

## 変更内容
事前にAI生成した架空の成人4人の顔を使用します。試行中に新しく画像を生成するAPI連携は含みません。MediaPipeで顔特徴点と相対奥行きを推定し、顔幅160mmの仮定で顔前面メッシュを生成。ブラシの向きを曲面の法線へ合わせ、実際の先端が表面8mm以内に来た時のみ色を重ねます。参考画像に寄せたローズ色と黒・銀のコンパクトを配置します。正確なDiorの商品CADや色番号は再現していません。

## 検証状況
Python構文と、合成メッシュ100点で座標投影・法線・ブラシ姿勢を確認済み。MediaPipeとPyBulletを含む全体実行は未確認です。短い確認セルから実行してください。

## 限界
これは顔前面の推定3D表面で、頭全体の精密モデルではありません。顔とブラシは可視化用で接触力を計算しません。皮膚変形・圧力制御・衝突回避は未実装。曲面に沿ったロボット動作の概念実証です。

## 評価
固定方式は最初の顔に合わせた軌道を使い、AI方式は各顔に合わせます。各顔・位置について両方式を試します。独立した正解形状がないので、検出精度は報告しません。記録するのは追従誤差、AI参照軌道との差、表面近傍フレーム率です。最初の顔は両方式が同条件です。旧平面版の6.45mm・83.5%は今回の結果に流用しないでください。

## 生成物
results/makeup_3d_demo.mp4、results/results.json、results/report_draft.md、4つのOBJモデルと画像。レポートは原稿で、PDF化は動作・数値確認後に行います。課題提出は別途必要です。

## 参考
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
https://github.com/bulletphysics/bullet3
画像生成・コード作成支援: ChatGPT。

## 動画
https://youtu.be/VlaPdqWlWuc

## 実行環境の修正
Colabから引き継ぐMatplotlibの描画設定をAggに固定し、専用環境でのimportエラーを回避します。結果表示セルでもprojectを定義しています。

## ローカル実行（Python 3.11）
```bash
python -m pip install -r requirements.txt
python makeup_3d.py --atlas portraits.png --smoke
python makeup_3d.py --atlas portraits.png
```
初回はMediaPipeモデルをGoogleの配布元からダウンロードします。結果はresults/に保存されます。
