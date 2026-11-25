# ReFuGe: Feature Generation for Prediction Tasks on Relational Databases with LLM Agents

[English](README.md) | **한국어**

ReFuGe **(Feature Generation for Prediction Tasks on Relational Databases with LLM Agents)**의 공식 구현 코드입니다.

**ACM WWW 2026** 게재 승인

[![arXiv](https://img.shields.io/badge/arXiv-2601.17735-b31b1b.svg)](https://arxiv.org/abs/2601.17735)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

### 🛠️ 요구 사항

ReFuGe는 벤치마크 데이터셋과 태스크 정의를 불러오기 위해 [RelBench](https://github.com/snap-stanford/relbench)를 사용합니다. 기본 Claude 백엔드는 `ANTHROPIC_API_KEY`를 사용합니다.

OpenAI 모델과 OpenAI 호환 서버도 지원합니다 ([다른 LLM 백본 사용하기](#-다른-llm-백본-사용하기) 참고).

### 🚀 실행 방법

```bash
git clone https://github.com/K-Kyungho/REFUGE.git
cd REFUGE
pip install -r requirements.txt
export ANTHROPIC_API_KEY="YOUR_API_KEY"

python run.py --task f1-dnf
```

사용 가능한 태스크: `f1-dnf`, `f1-top3`, `event-repeat`, `event-ignore`,
`avito-clicks`, `avito-visits`, `amazon-churn`.

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--n_iter` | 3 | 최대 반복 횟수 (한 반복에서 채택된 피처가 없으면 조기 종료) |
| `--models` | 태스크별 | 쉼표로 구분한 플래너 모델 ID 3개 |
| `--judge_model` | `claude-sonnet-4-20250514` | 9개 계획 중 3개를 선택하고 피처 코드를 작성 |
| `--provider` | `anthropic` | LLM 백엔드: `anthropic` 또는 `openai` |
| `--base_url` | 없음 | OpenAI 호환 서버의 커스텀 엔드포인트 (예: vLLM) |
| `--api_key` | `$ANTHROPIC_API_KEY` / `$OPENAI_API_KEY` | API 키; 기본값은 선택한 provider의 환경 변수 |
| `--temp` | 0.2 | 기본 temperature (플래너 *k*는 `temp + 0.2k` 사용) |
| `--delta` | 0.0 | val AUROC ≥ best + delta 인 경우에만 피처를 채택 |
| `--data_dir` | `data/` | LLM에 제공되는 스키마, 통계, 태스크 파일이 있는 디렉터리 |
| `--out_dir` | `outputs/<dataset>/<task>` | 결과 저장 경로 |
| `--seed` | 없음 | 평가기의 train 서브샘플링 시드 |

출력물 (`--out_dir`):

```
<task>-{train,val,test}.csv   채택된 모든 피처가 추가된 태스크 테이블
features/                     생성된 코드가 만든 피처별 CSV
generated_code/iter<i>.py     반복마다 LLM이 작성한 피처 코드
```

### 🔌 다른 LLM 백본 사용하기

논문 실험은 Claude를 사용했지만, 백본은 변경할 수 있습니다. 모든 LLM 호출은
`refuge/llm.py`의 `ask_model`을 거치며, `--provider`로 백엔드를 선택합니다.

**OpenAI 모델**

```bash
pip install openai
export OPENAI_API_KEY="YOUR_API_KEY"

python run.py --task f1-dnf --provider openai \
    --models gpt-4.1,gpt-4.1-mini,gpt-4o --judge_model gpt-4.1
```

**오픈 웨이트 모델 (또는 OpenAI 호환 API)** — vLLM, SGLang, Ollama 등:

```bash
vllm serve Qwen/Qwen3-32B --port 8000

python run.py --task f1-dnf --provider openai --base_url http://localhost:8000/v1 --api_key EMPTY \
    --models Qwen/Qwen3-32B,Qwen/Qwen3-32B,Qwen/Qwen3-32B --judge_model Qwen/Qwen3-32B
```

참고:
- `--provider openai`를 사용할 때는 태스크별 기본값이 Claude 모델 ID이므로
  `--models`(플래너 ID 3개)와 `--judge_model`을 반드시 지정해야 합니다.
- 플래너 *k*는 다양한 계획을 얻기 위해 temperature `temp + 0.2k`를 사용합니다. temperature를
  지원하지 않는 모델(예: OpenAI 추론 모델)은 경고와 함께 기본값으로 대체됩니다.
- judge가 실행 가능한 Python 코드를 작성하므로, `--judge_model`에는 코드 성능이 좋은 모델을 쓰는 것이 가장 중요합니다.
- 논문에 보고된 결과는 Claude로 얻은 것이며, 다른 백본에서는 성능이 다를 수 있습니다.
- 다른 provider를 추가하려면 `refuge/llm.py`의 `make_client`와 `ask_model`에 응답 텍스트를
  반환하는 분기를 추가하면 됩니다. 파이프라인의 다른 부분은 수정할 필요가 없습니다.

### 📁 코드 구조

```
run.py              CLI 진입점
refuge/
  tasks.py          태스크별 설정 (엔티티 테이블/키, 타깃, 평가기 컬럼, 모델)
  pipeline.py       메인 루프: 테이블 → 컬럼 선택 → 계획 9개 → judge → 코드 → 검증 필터
  prompts.py        각 단계의 프롬프트 템플릿
  evaluate.py       LightGBM 검증 AUROC 체크
  parsing.py        계획, judge JSON, 생성 코드 파싱
  llm.py            LLM API 래퍼 (Anthropic / OpenAI 호환)
data/rel-*/         LLM에 제공되는 스키마, 컬럼 통계, 태스크 설명
```

태스크를 추가하려면 `refuge/tasks.py`에 `TaskConfig` 항목을 추가하세요 (rel-hm, rel-stack,
rel-trial의 스키마/통계/태스크 파일은 이미 `data/`에 있습니다).

### 📝 인용

ReFuGe가 연구에 도움이 되었다면 다음 논문을 인용해 주세요:

```bibtex
@inproceedings{kim2026refuge,
  title     = {ReFuGe: Feature Generation for Prediction Tasks on Relational Databases with LLM Agents},
  author    = {Kim, Kyungho and Lee, Geon and Kim, Juyeon and Choi, Dongwon and Kang, Shinhwan and Shin, Kijung},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  publisher = {ACM},
  eprint    = {2601.17735},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2601.17735}
}
```
