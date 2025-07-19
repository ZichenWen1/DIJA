<div align="center">
  <h1 style="display: inline-block; margin: 0;">🎭 The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs</h1>
</div>

<h4 align="center"> 

[Zichen Wen](https://scholar.google.com/citations?user=N-aPFvEAAAAJ&hl=zh-CN)<sup>1,2</sup>,
[Jiashu Qu](https://tree985.github.io/)<sup>2</sup>,
[Dongrui Liu](https://shenqildr.github.io/)<sup>2*</sup>,
Zhiyuan Liu<sup>1,2</sup>,
Ruixi Wu<sup>1,2</sup>, 
Yicun Yang<sup>1</sup>,
Xiangqi Jin<sup>1</sup>, <br>
Haoyun Xu<sup>1</sup>,
[Xuyang Liu](https://xuyang-liu16.github.io/)<sup>1</sup>,
[Weijia Li](https://liweijia.github.io/)<sup>3,2</sup>,
[Chaochao Lu](https://causallu.com/)<sup>2</sup>,
[Jing Shao](https://amandajshao.github.io/)<sup>2</sup>,
[Conghui He](https://conghui.github.io/)<sup>2✉</sup>,
[Linfeng Zhang](http://www.zhanglinfeng.tech/)<sup>1✉</sup>,


<sup>1</sup>EPIC Lab, Shanghai Jiao Tong University, <sup>2</sup>Shanghai AI Laboratory, <br>
<sup>3</sup>Sun Yat-sen University

✉Corresponding authors, *Project lead

</h4>

<div align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2507.11097-AD1C18.svg?logo=arXiv)](https://arxiv.org/pdf/2507.11097)
[![](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FZichenWen1%2FDART&count_bg=%23C25AE6&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitor&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub issues](https://img.shields.io/github/issues/ZichenWen1/DIJA?color=critical&label=Issues)](https://github.com/ZichenWen1/DIJA/issues)
[![GitHub Stars](https://img.shields.io/github/stars/ZichenWen1/DIJA?style=social)](https://github.com/ZichenWen1/DIJA/stargazers)
</div>

## 📰 News
* **`2025.07.16`** 🤗🤗 We release our latest work [DIJA](https://arxiv.org/pdf/2507.11097), the **first investigation** into the safety issues of dLLMs. [Code](https://github.com/ZichenWen1/DIJA) is available!

## 👀 Overview

- 💥 This is the **first investigation** into the safety issues of dLLMs. We identify and characterize a novel attack pathway against dLLMs, rooted in their bidirectional and parallel decoding mechanisms.
- 💥 We propose DIJA, an automated jailbreak attack pipeline that transforms vanilla jailbreak prompts into interleaved text-mask jailbreak prompts capable of eliciting harmful completions on dLLMs.
- 💥 We conduct comprehensive experiments demonstrating the effectiveness of DIJA across multiple dLLMs compared with existing attack methods, highlighting critical gaps in current alignment strategies and exposing urgent security vulnerabilities in existing dLLM architectures that require immediate addressing.

<p align='center'>
<img src='./assets/attack_cases.jpg' alt='mask' width='900px'>
</p>

## 📊 Performance
- 🎯 DIJA achieves the **highest** ASR-k across all benchmarks, indicating that dLLMs are highly unlikely to refuse answering dangerous or sensitive topics under the DIJA attack.
- 🎯 For the more secure Dream-Instruct, DIJA achieves an improvement of up to 78.5% in ASR-e on JailbreakBench over the best baseline, ReNeLLM, and a 37.7% improvement in StrongREJECT score.

<p align='center'>
<img src='./assets/harmbench_exp.jpg' alt='mask' width='900px'>
</p>

<p align='center'>
<img src='./assets/jailbreakbench_exp.jpg' alt='mask' width='900px'>
</p>

<p align='center'>
<img src='./assets/strongreject_exp.jpg' alt='mask' width='900px'>
</p>


## 🛠 Preparation
1. Clone this repository.

```bash
  git clone https://github.com/ZichenWen1/DIJA
  cd DIJA
```
   
2. Install models
   
```bash
  cd hf_models && bash model_download.sh
```

3. Environment setup

```bash
  conda create -n DIJA python=3.10 -y
  conda activate DIJA
  pip install -r requirements.txt
```

## 🧪 Usage and Evaluation

### Parameters
- [Version]: You can set the version number for this run
- [Defense_method]: Choose whether to apply defense during the attack. Options: None, Self-reminder, RPO
- [Victim_model]: Select the targeted diffusion LLM. Options: llada_instruct, llada_1.5, dream_instruct, mmada_mixcot

### HarmBench evaluation
```bash
  # Interleaved mask-text prompt construction
  cd run_harmbench
  bash refine_prompt/run_refine.sh [Version]

  # Jailbreak attack and evaluation
  bash eval_harmbench.sh DIJA [Defense_method] [Victim_model] [Version]
```

### JailbreakBench evaluation
```bash
  # Interleaved mask-text prompt construction
  cd run_jailbreakbench
  bash refine_prompt/run_refine.sh [Version]

  # Jailbreak attack and evaluation
  bash eval_jailbreakbench.sh DIJA [Defense_method] [Victim_model] [Version]
```

### StrongREJECT evaluation
```bash
  # Interleaved mask-text prompt construction
  cd run_strongreject
  bash refine_prompt/run_refine.sh [Version]

  # Jailbreak attack and evaluation
  bash eval_strongreject.sh DIJA [Defense_method] [Victim_model] [Version]
```


## 📌 TODO
- [x] Release Inference and Evaluation Code
- [ ] Support DiffuCoder, Dream-Coder 

## 🔑 License

This project is released under the [Apache 2.0 license](LICENSE).

## 📍 Citation

Please consider citing our paper in your publications if our works help your research.
```bibtex
@article{wen2025devil,
  title={The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs},
  author={Wen, Zichen and Qu, Jiashu and Liu, Dongrui and Liu, Zhiyuan and Wu, Ruixi and Yang, Yicun and Jin, Xiangqi and Xu, Haoyun and Liu, Xuyang and Li, Weijia and others},
  journal={arXiv preprint arXiv:2507.11097},
  year={2025}
}
```

## 👍 Acknowledgments

### Diffusion LLMs  
We would like to express our sincere gratitude to the open-source contributions from the teams behind [LLaDA](https://github.com/ML-GSAI/LLaDA), [LLaDA-1.5](https://github.com/ML-GSAI/LLaDA-1.5), [Dream](https://github.com/DreamLM/Dream), and [MMaDA](https://github.com/Gen-Verse/MMaDA).

### Jailbreak Benchmarks  
We are deeply appreciative of the open-source efforts by the developers of [HarmBench](https://github.com/centerforaisafety/HarmBench), [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench), and [StrongREJECT](https://github.com/dsbowen/strong_reject).


