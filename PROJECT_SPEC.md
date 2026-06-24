# Project Specification: OSCAR-LIC

## 1. Problem statement

一般的なlearned image compressionはPSNR、MS-SSIM、LPIPS等を最適化しますが、低bitrateで消失する細いstroke、文字間隔、hole、diacriticは、面積が小さい一方でOCRに大きな意味変化を与えます。既存のtext-aware手法は主に、単一OCR loss、text maskによるROI weighting、文字列side-channel＋生成描画に分類できます。

本projectは、次の問題を扱います。

> 文字列そのものを送信せず、未知OCRへも転移するvisual evidenceを、全side information込みの限られたbit budgetで選択的に保存できるか。

## 2. Primary hypothesis

### H1: Marginal OCR utility per bit

text existence maskやdetector confidenceより、追加bitによるOCR loss減少を実bit増分で割ったcounterfactual utilityを用いる方が、同一total bppで低いheld-out CER/WERを得る。

\[
u_i(S)=\frac{L_{OCR}(\hat{x}_S)-L_{OCR}(\hat{x}_{S\cup\{i\}})}{R(S\cup\{i\})-R(S)+\epsilon}
\]

ここで `i` はlatent tile、channel group、bit-plane等の送信単位、`S` は既送信集合です。

## 3. Secondary hypotheses

- **H2:** multi-teacher feature/logit supervisionとteacher dropoutはsingle-teacher lossよりheld-out OCRへ転移する。
- **H3:** pixel residualではなくstroke/topology-oriented enhancement latentは、同rateでsmall-text CERを改善する。
- **H4:** base＋sparse enhancementのscalable streamは、固定一層codecより用途別のrate allocationを容易にする。
- **H5:** text ROI限定のone-step generative refinerは、multi-step full-image diffusionより高速で、deterministic fidelityを保てる。
- **H6:** text enhancement streamへのunequal error protectionは、同じFEC overheadで一様保護よりCER degradationを抑える。

## 4. Core contribution boundary

主論文の中心貢献は以下の三つです。

1. OCR-model-invariant multi-teacher objective
2. counterfactual utility-per-bit allocation
3. plaintext-free scalable text enhancement stream

one-step diffusion、compressed-domain OCR、privacy、RAW連携、channel robustnessは重要ですが、中心貢献を曖昧にしないためoptional extensionまたはfollow-upとして扱います。

## 5. Tracks and fairness

### A. Pure-image coding

入力は画像のみ、出力bitstreamから画像を復元。文字列・OCR token・box座標のlossless送信は禁止。OSCAR-LICの主結果。

### B. Semantic-assisted coding

文字列・layout metadataを送信可能。PICD等と比較。ただしside bitrate、OCR encoder latency、privacyを全計上。

### C. Compressed-domain machine coding

latentから直接OCR/KIE/DocVQAを実行。RGB decoderを経由する結果と別に報告。

## 6. Proposed system

### 6.1 Base codec

最初のimplementation targetは MLIC++。理由は、公式code、checkpoint、実bitstream path、multi-reference entropy modelが利用できることです。DCAEを第二baselineとし、dictionary priorをglyph/stroke prototypeへ拡張する可能性を検証します。

### 6.2 Utility predictor

base latent、hyperlatent、低コストtextness、reconstruction uncertaintyから、各候補unitのexpected `ΔOCR-loss / Δbits` と不確実性を予測します。deployment時に重いOCRを実行しないよう、offline multi-OCR oracleからdistillationします。

### 6.3 Enhancement stream

候補unitは初期実装では `8×8 latent tile × channel group` とします。gate自体はentropy codingし、無料のmaskとして扱いません。streamはbaseなしでは意味を持たないconditional enhancementです。

### 6.4 Representation candidates

優先順位:

1. learned feature residual
2. signed distance / edge orientation auxiliary targetを持つlatent
3. skeleton/topology latent
4. VQ glyph token

最初から複数表現を混ぜず、同じrate budgetで比較します。

## 7. Loss

\[
\begin{aligned}
L=&R_{all}+\lambda_D L_D+\lambda_{inv}L_{OCR-inv}+\lambda_{seq}L_{seq}\\
&+\lambda_{top}L_{top}+\lambda_U L_U+\lambda_B L_{budget}
+\lambda_{rob}L_{rob}
\end{aligned}
\]

`R_all` はbase、hyper、gate、text、metadata、CRC/FECを含みます。詳細は `docs/losses.md`。

## 8. Primary outcomes

- BD-Rate@CER
- BD-Rate@WER
- held-out worst-OCR CER
- exact word accuracy
- total bpp
- text/non-text LPIPSまたはDISTS
- encoder/decoder latencyとpeak memory

## 9. Required datasets

最低限、次の3 domainを持つこと。

- scene text
- screen/UI content
- document/receipt/form

一般画像を混ぜ、文字以外の画質崩壊を防ぎます。dataset詳細とlicenseは `data/registry.yaml`。

## 10. Success criteria

### Minimum publishable target

- strong pure-image text-aware baselineに対しheld-out OCRのBD-Rate@CERまたはWERを15%以上改善
- 3 domain中2 domain以上で改善
- mask/index/header込みtotal bppで改善
- random-utility、uniform text allocation、detector confidenceを上回る
- non-text画質を実用上大きく悪化させない
- OCRなしdeployment encoderを示す

### Stretch target

- 25–35% BD-rate改善
- enhancement side rateがtotalの5–15%範囲で高効率
- one-step ROI refiner込みsub-second GPU decode
- bit error時のCER増分を一様保護より50%削減

数値目標は達成済み結果ではありません。

## 11. No-Go conditions

- oracle utilityが同一候補集合のmask allocationを上回らない
- improvementがtraining OCRだけで消える
- side bits込みで優位性が消える
- synthetic-to-realで再現しない
- generatorが数字・固有名詞をhallucinateする
- full OCRをencoderで実行しないと性能が成立しない

## 12. Non-goals for the first paper

- lossless compression
- video compression
- 全言語scriptの完全対応
- RAW、security、MLLM、diffusionの同時統合
- proprietary OCR APIを主評価にすること
- OCR outputを正解文字列として送ること

## 13. Target venues

- CVPR / ICCV / ECCV: 視覚・ICM・OCR benchmarkを強調
- ICLR / NeurIPS: utility allocationとmodel invarianceの一般性を強調
- DCC / ICIP: coding syntax、rate accounting、reproducibilityを強調
