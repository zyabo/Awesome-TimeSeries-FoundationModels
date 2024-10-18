# Awesome-TimeSeries-FoundationModels

Awesome resources and papers about leveraging Foundation Models for Time Series Analysis.

## 🍭 News

- **Update Logs**
  - **2024/10/18** Add 10 papers



## 🍻Time Series Foundation Models

|      | Year | Model Name              | Paper Title                                                  | Paper URL                                 | Code/API/HuggingFace                                         | Authors                                                      |
| ---- | ---- | ----------------------- | ------------------------------------------------------------ | ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | 2023 | TimeGPT-1               | TimeGPT-1                                                    | [Paper](https://arxiv.org/abs/2310.03589) | [API](https://github.com/Nixtla/nixtla)                      | [Nixtla](https://www.nixtla.io/)                             |
| 2    | 2023 | Lag-Llama               | Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting | [Paper](https://arxiv.org/abs/2310.08278) | [Code](https://github.com/time-series-foundation-models/lag-llama), [Hugging Face](https://huggingface.co/time-series-foundation-models/Lag-Llama) | [Kashif Rasul](https://arxiv.org/search/cs?searchtype=author&query=Rasul,+K),et al. |
| 3    | 2023 | TimesFM                 | A decoder-only foundation model for time-series forecasting  | [Paper](https://arxiv.org/abs/2310.10688) | [Code](https://github.com/google-research/timesfm), [Hugging Face](https://huggingface.co/google/timesfm-1.0-200m) | Google（Abhimanyu Das , et al.）                             |
| 4    | 2024 | Tiny Time Mixers (TTMs) | Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series | [Paper](https://arxiv.org/abs/2401.03955) | [Hugging Face](https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1) | [Vijay Ekambaram](https://arxiv.org/search/cs?searchtype=author&query=Ekambaram,+V),et al. |
| 5    | 2024 | Moirai                  | Unified Training of Universal Time Series Forecasting Transformers | [Paper](https://arxiv.org/abs/2402.02592) | [Code](https://github.com/SalesforceAIResearch/uni2ts), [Hugging Face](https://huggingface.co/Salesforce/moirai-1.0-R-large) | [Gerald Woo](https://arxiv.org/search/cs?searchtype=author&query=Woo,+G), et al. |
| 6    | 2024 | Chronos                 | Chronos: Learning the Language of Time Series by Amazon      | [Paper](https://arxiv.org/abs/2403.07815) | [Code](https://github.com/amazon-science/chronos-forecasting), [Hugging Face](https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444) | [Abdul Fatir Ansari](https://arxiv.org/search/cs?searchtype=author&query=Ansari,+A+F),et al. |
| 7    | 2024 | Timer                   | Timer: Generative Pre-trained Transformers Are Large Time Series Models | [Paper](https://arxiv.org/abs/2402.02368) | [Code](https://github.com/thuml/Large-Time-Series-Model?tab=readme-ov-file) | [Yong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+Y), et al. |
| 8    | 2024 | TOTEM                   | TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis | [Paper](https://arxiv.org/abs/2402.16412) | [Code](https://github.com/SaberaTalukder/TOTEM)              | [Sabera Talukder](https://arxiv.org/search/cs?searchtype=author&query=Talukder,+S),  et al. |



## 🍪Review Papers

### 1.A Survey of Time Series Foundation Models: Generalizing Time Series Representation with Large Language Model

[Paper](https://arxiv.org/pdf/2405.02358)

Abstract

> Time series data are ubiquitous across various domains, making time series analysis critically important. Traditional time series models are task-specific, featuring singular functionality and limited generalization capacity. Recently, large language foundation models have unveiled their remarkable capabilities for cross-task transferability, zero-shot/few-shot learning, and decision-making explainability. This success has sparked interest in the exploration of foundation models to solve multiple time series challenges simultaneously. There are two main research lines, namely pre-training foundation models from scratch for time series and adapting large language foundation models for time series. They both contribute to the development of a unified model that is highly generalizable, versatile, and comprehensible for time series analysis. This survey offers a 3E analytical framework for comprehensive examination of related research. Specifically, we examine existing works from three dimensions, namely Effectiveness, Efficiency and Explainability. In each dimension, we focus on discussing how related works devise tailored solution by considering unique challenges in the realm of time series. Furthermore, we provide a domain taxonomy to help followers keep up with the domain-specific advancements. In addition, we introduce extensive resources to facilitate the field's development, including datasets, open-source, time series libraries. A GitHub repository is also maintained for resource updates ([this https URL](https://github.com/start2020/Awesome-TimeSeries-LLM-FM)).



### 2.Foundation Models for Time Series Analysis: A Tutorial and Survey

[Paper](https://arxiv.org/abs/2403.14735)

Abstract

> Time series analysis stands as a focal point within the data mining community, serving as a cornerstone for extracting valuable insights crucial to a myriad of real-world applications. Recent advances in Foundation Models (FMs) have fundamentally reshaped the paradigm of model design for time series analysis, boosting various downstream tasks in practice. These innovative approaches often leverage pre-trained or fine-tuned FMs to harness generalized knowledge tailored for time series analysis. This survey aims to furnish a comprehensive and up-to-date overview of FMs for time series analysis. While prior surveys have predominantly focused on either application or pipeline aspects of FMs in time series analysis, they have often lacked an in-depth understanding of the underlying mechanisms that elucidate why and how FMs benefit time series analysis. To address this gap, our survey adopts a methodology-centric classification, delineating various pivotal elements of time-series FMs, including model architectures, pre-training techniques, adaptation methods, and data modalities. Overall, this survey serves to consolidate the latest advancements in FMs pertinent to time series analysis, accentuating their theoretical underpinnings, recent strides in development, and avenues for future exploration.	





## 🍉 More Details

### 1.TimeGPT-1

#### Abstract

> In this paper, we introduce TimeGPT, the first foundation model for time series, capable of generating accurate predictions for diverse datasets not seen during training. We evaluate our pre-trained model against established statistical, machine learning, and deep learning methods, demonstrating that TimeGPT zero-shot inference excels in performance, efficiency, and simplicity. Our study provides compelling evidence that insights from other domains of artificial intelligence can be effectively applied to time series analysis. We conclude that large-scale time series models offer an exciting opportunity to democratize access to precise predictions and reduce uncertainty by leveraging the capabilities of contemporary advancements in deep learning.
>



### 2.Lag-Llama

#### Abstract

> Over the past years, foundation models have caused a paradigm shift in machine learning due to their unprecedented capabilities for zero-shot and few-shot generalization. However, despite the success of foundation models in modalities such as natural language processing and computer vision, the development of foundation models for time series forecasting has lagged behind. We present Lag-Llama, a general-purpose foundation model for univariate probabilistic time series forecasting based on a decoder-only transformer architecture that uses lags as covariates. Lag-Llama is pretrained on a large corpus of diverse time series data from several domains, and demonstrates strong zero-shot generalization capabilities compared to a wide range of forecasting models on downstream datasets across domains. Moreover, when fine-tuned on relatively small fractions of such previously unseen datasets, Lag-Llama achieves state-of-the-art performance, outperforming prior deep learning approaches, emerging as the best general-purpose model on average. Lag-Llama serves as a strong contender to the current state-of-art in time series forecasting and paves the way for future advancements in foundation models tailored to time series data.
>



### 3.TimesFM

#### Abstract

> Motivated by recent advances in large language models for Natural Language Processing (NLP), we design a time-series foundation model for forecasting whose out-of-the-box zero-shot performance on a variety of public datasets comes close to the accuracy of state-of-the-art supervised forecasting models for each individual dataset. Our model is based on pretraining a patched-decoder style attention model on a large time-series corpus, and can work well across different forecasting history lengths, prediction lengths and temporal granularities.
>



### 4.TTMs

#### Abstract

> Large pre-trained models excel in zero/few-shot learning for language and vision tasks but face challenges in multivariate time series (TS) forecasting due to diverse data characteristics. Consequently, recent research efforts have focused on developing pre-trained TS forecasting models. These models, whether built from scratch or adapted from large language models (LLMs), excel in zero/few-shot forecasting tasks. However, they are limited by slow performance, high computational demands, and neglect of cross-channel and exogenous correlations. To address this, we introduce Tiny Time Mixers (TTM), a compact model (starting from 1M parameters) with effective transfer learning capabilities, trained exclusively on public TS datasets. TTM, based on the light-weight TSMixer architecture, incorporates innovations like adaptive patching, diverse resolution sampling, and resolution prefix tuning to handle pre-training on varied dataset resolutions with minimal model capacity. Additionally, it employs multi-level modeling to capture channel correlations and infuse exogenous signals during fine-tuning. TTM outperforms existing popular benchmarks in zero/few-shot forecasting by (4-40\%), while reducing computational requirements significantly. Moreover, TTMs are lightweight and can be executed even on CPU-only machines, enhancing usability and fostering wider adoption in resource-constrained environments. Model weights for our initial variant (TTM-Q) are available at [this https URL](https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1). Model weights for more sophisticated variants (TTM-B, TTM-E, and TTM-A) will be shared soon. The source code for TTM can be accessed at [this https URL](https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/tinytimemixer).
>



### 5.Moirai

#### Abstract

> Deep learning for time series forecasting has traditionally operated within a one-model-per-dataset framework, limiting its potential to leverage the game-changing impact of large pre-trained models. The concept of universal forecasting, emerging from pre-training on a vast collection of time series datasets, envisions a single Large Time Series Model capable of addressing diverse downstream forecasting tasks. However, constructing such a model poses unique challenges specific to time series data: i) cross-frequency learning, ii) accommodating an arbitrary number of variates for multivariate time series, and iii) addressing the varying distributional properties inherent in large-scale data. To address these challenges, we present novel enhancements to the conventional time series Transformer architecture, resulting in our proposed Masked Encoder-based Universal Time Series Forecasting Transformer (Moirai). Trained on our newly introduced Large-scale Open Time Series Archive (LOTSA) featuring over 27B observations across nine domains, Moirai achieves competitive or superior performance as a zero-shot forecaster when compared to full-shot models. Code, data, and model weights can be found at [this https URL](https://github.com/SalesforceAIResearch/uni2ts).
>



### 6.Chronos

#### Abstract

> We introduce Chronos, a simple yet effective framework for pretrained probabilistic time series models. Chronos tokenizes time series values using scaling and quantization into a fixed vocabulary and trains existing transformer-based language model architectures on these tokenized time series via the cross-entropy loss. We pretrained Chronos models based on the T5 family (ranging from 20M to 710M parameters) on a large collection of publicly available datasets, complemented by a synthetic dataset that we generated via Gaussian processes to improve generalization. In a comprehensive benchmark consisting of 42 datasets, and comprising both classical local models and deep learning methods, we show that Chronos models: (a) significantly outperform other methods on datasets that were part of the training corpus; and (b) have comparable and occasionally superior zero-shot performance on new datasets, relative to methods that were trained specifically on them. Our results demonstrate that Chronos models can leverage time series data from diverse domains to improve zero-shot accuracy on unseen forecasting tasks, positioning pretrained models as a viable tool to greatly simplify forecasting pipelines.
>





### 7.Timer

#### Abstract

> Deep learning has contributed remarkably to the advancement of time series analysis. Still, deep models can encounter performance bottlenecks in real-world data-scarce scenarios, which can be concealed due to the performance saturation with small models on current benchmarks. Meanwhile, large models have demonstrated great powers in these scenarios through large-scale pre-training. Continuous progress has been achieved with the emergence of large language models, exhibiting unprecedented abilities such as few-shot generalization, scalability, and task generality, which are however absent in small deep models. To change the status quo of training scenario-specific small models from scratch, this paper aims at the early development of large time series models (LTSM). During pre-training, we curate large-scale datasets with up to 1 billion time points, unify heterogeneous time series into single-series sequence (S3) format, and develop the GPT-style architecture toward LTSMs. To meet diverse application needs, we convert forecasting, imputation, and anomaly detection of time series into a unified generative task. The outcome of this study is a Time Series Transformer (Timer), which is generative pre-trained by next token prediction and adapted to various downstream tasks with promising capabilities as an LTSM. Code and datasets are available at: [this https URL](https://github.com/thuml/Large-Time-Series-Model).
>



### 8.TOTEM

#### Abstract

> The field of general time series analysis has recently begun to explore unified modeling, where a common architectural backbone can be retrained on a specific task for a specific dataset. In this work, we approach unification from a complementary vantage point: unification across tasks and domains. To this end, we explore the impact of discrete, learnt, time series data representations that enable generalist, cross-domain training. Our method, TOTEM, or TOkenized Time Series EMbeddings, proposes a simple tokenizer architecture that embeds time series data from varying domains using a discrete vectorized representation learned in a self-supervised manner. TOTEM works across multiple tasks and domains with minimal to no tuning. We study the efficacy of TOTEM with an extensive evaluation on 17 real world time series datasets across 3 tasks. We evaluate both the specialist (i.e., training a model on each domain) and generalist (i.e., training a single model on many domains) settings, and show that TOTEM matches or outperforms previous best methods on several popular benchmarks. The code can be found at: [this https URL](https://github.com/SaberaTalukder/TOTEM).
