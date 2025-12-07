# 멜로디 생성기 (Melody Generator) - DQN 기반

이 프로젝트는 강화 학습(Reinforcement Learning)의 한 종류인 DQN(Deep Q-Network)을 활용하여 MIDI 코드 진행에 어울리는 멜로디를 자동으로 생성하는 모델을 구현합니다.

## 1. 프로젝트 개요

주어진 음악의 코드 진행(화음, 박자, 속도 정보)에 기반하여 가장 적절한 멜로디 음(pitch)과 세기(velocity)를 선택하는 에이전트를 학습시킵니다. 학습된 에이전트는 음악 이론에 부합하며 자연스러운 멜로디 라인을 생성하는 것을 목표로 합니다.

## 2. 데이터 준비 및 전처리

### 2.1 데이터 다운로드
Kaggle `imsparsh/lakh-midi-clean` 데이터셋을 `kagglehub` 라이브러리를 사용하여 다운로드합니다.

### 2.2 300개 MIDI 파일 샘플링
너무 많은 데이터를 한꺼번에 처리하는 것을 방지하기 위해, 다운로드된 데이터셋에서 무작위로 300개의 MIDI 파일을 선택하여 `midi_sample_300` 디렉토리에 복사합니다.

### 2.3 MIDI 데이터 전처리
각 MIDI 파일에서 멜로디 생성을 위한 중요한 정보(beat, pitch, velocity)를 추출합니다.

-   **`extract_chords_by_beat(midi_path)`**: MIDI 파일에서 BPM, 각 beat의 시작 시간, 그리고 각 beat에 해당하는 화음(chords)과 해당 화음의 평균 velocity 정보를 추출합니다. 화음은 해당 beat 동안 울리는 모든 pitch의 집합입니다.
-   **`make_64beat_windows(chords_by_beat, velocities_by_beat)`**: 추출된 beat 단위의 화음/velocity 시퀀스를 기반으로, 길이가 64 beat인 슬라이딩 윈도우(segments)들을 생성합니다. 이 64-beat 윈도우가 강화 학습 환경의 한 에피소드에 해당합니다.

### 2.4 전처리된 데이터 저장
처리된 모든 64-beat 윈도우들은 `chord_windows` 리스트에 저장되며, 이 리스트는 `data.pkl` 파일로 직렬화(pickle)되어 저장됩니다. 이 파일은 이후 멜로디 환경 구성에 사용됩니다.

## 3. 멜로디 환경 구성 (`MelodyEnv`)

멜로디 생성 에이전트가 상호작용할 강화 학습 환경인 `MelodyEnv` 클래스를 정의합니다. 이 환경은 다음을 포함합니다:

-   **`__init__`**: `chord_windows`, `pitch_vocab`(사용 가능한 모든 pitch), `vel_bins`(사용 가능한 velocity 단계), `max_steps`(에피소드 길이, 64 beat), `history_len`(state에 포함될 과거 멜로디 길이)를 초기화합니다.
-   **`reset()`**: 새로운 에피소드를 시작할 때 호출됩니다. 무작위로 64-beat 코드 진행을 선택하고, 환경의 상태를 초기화(멜로디 히스토리, 현재 스텝 등)한 후 초기 상태를 반환합니다.
-   **`_decode_action(action_idx)`**: 에이전트가 선택한 action index를 실제 pitch와 velocity 값으로 변환합니다.
-   **`_reward(pitch, vel)`**: 에이전트가 생성한 멜로디 음(pitch, vel)에 대한 보상을 계산합니다. 보상은 음악 이론(코드톤 여부, 이전 음과의 관계, velocity의 적절성, 음정 도약 등)에 기반하여 설계되었습니다.
-   **`_build_state()`**: 현재 시점의 환경 상태(state)를 구성합니다. 이는 최근 `history_len`개의 멜로디 음 정보와 현재 beat의 코드 진행 정보로 이루어집니다.
-   **`step(action_idx)`**: 에이전트의 액션을 받아 환경을 한 스텝 진행시키고, 다음 상태, 보상, 에피소드 종료 여부(`done`)를 반환합니다.

### 💡 **모델 실행 시 주의사항 (데이터 로드)**

모델을 테스트하기 위해 `data.pkl`을 불러오는 코드에서 경로를 수정해야 합니다. 아래 코드 셀에 제시된 바와 같이, `data.pkl` 대신 제공된 `music_300.pkl` 파일의 경로를 넣어주세요.

```python
# 기존 코드:
# with open("/content/drive/MyDrive/data/data.pkl", "rb") as f:
#     data_dict = pickle.load(f)

# music_300.pkl을 사용하는 경우:
with open("YOUR_MUSIC_300_PKL_PATH/music_300.pkl", "rb") as f:
    data_dict = pickle.load(f)
```
`YOUR_MUSIC_300_PKL_PATH` 부분에 `music_300.pkl` 파일이 저장된 실제 경로를 입력해야 합니다.

## 4. DQN 모델 및 리플레이 버퍼

### 4.1 DQN (`DQN` Class)

에이전트의 정책을 학습하는 신경망입니다. `state_dim`을 입력으로 받아 각 `n_actions`에 대한 Q-값을 출력하는 간단한 Feed-forward 네트워크로 구성됩니다.

### 4.2 리플레이 버퍼 (`ReplayBuffer` Class)

에이전트의 경험(상태, 행동, 보상, 다음 상태, 종료 여부)을 저장하고, 학습 시 무작위로 샘플링하여 신경망 학습의 안정성을 높입니다.

## 5. 하이퍼파라미터 설정

DQN 학습을 위한 주요 하이퍼파라미터는 다음과 같습니다:

-   `batch_size`: 64
-   `gamma`: 0.5 (할인율)
-   `target_update_interval`: 1000 (타겟 네트워크 업데이트 주기)
-   `min_buffer_size`: 1000 (리플레이 버퍼 최소 크기)
-   `max_episodes`: 5000
-   `eps_start`: 1.0 (입실론-탐험 초기값)
-   `eps_end`: 0.05 (입실론-탐험 최종값)
-   `eps_decay_steps`: 50,000 (입실론 감소 스텝 수)

## 6. 학습 과정

각 에피소드마다 에이전트는 `epsilon-greedy` 전략에 따라 행동을 선택합니다. 선택된 행동에 따라 환경에서 보상과 다음 상태를 받고, 이 경험은 리플레이 버퍼에 저장됩니다. 리플레이 버퍼에 충분한 경험이 쌓이면, 버퍼에서 미니배치를 샘플링하여 DQN을 학습시킵니다. 일정 스텝마다 타겟 네트워크를 동기화하여 학습의 안정성을 확보합니다.

학습 진행 상황은 에피소드별 보상 그래프와 엡실론 감소 그래프로 시각화하여 확인할 수 있습니다.

## 7. 정책 평가

학습된 DQN 정책의 성능을 평가하기 위해 다음을 수행합니다:

-   **`run_episode_and_collect_metrics()`**: 한 에피소드를 실행하면서 총 보상, 코드톤에 포함된 음의 비율, 평균 음정 도약 크기, pitch 분산 등 다양한 음악 이론 기반 지표를 수집합니다.
-   **`evaluate_policy()`**: `n_episodes` 동안 정책을 실행하여 수집된 지표들의 평균을 계산하고, 이를 무작위 정책(`random_policy`)과 비교하여 학습 효과를 분석합니다.
-   **`play_episode_and_save_midi()`**: 특정 정책으로 한 에피소드를 실행하고, 생성된 멜로디와 코드 진행을 MIDI 파일로 저장하여 실제 음악으로 변환합니다. 이를 통해 학습된 정책이 생성한 멜로디의 음악적 품질을 직접 들어볼 수 있습니다.

### 7.1 평가 결과 분석

-   **에피소드 Return 분포**: 학습된 DQN 정책이 무작위 정책보다 훨씬 높은 평균 보상을 달성하며, 보상 분포가 양수 방향으로 이동함을 확인합니다.
-   **음악 이론 기반 지표 비교**: 코드톤 비율(`avg_ratio_in_chord`), 평균 음정 도약(`avg_leap`), Pitch 분산(`avg_pitch_variance`) 등의 지표를 통해 DQN이 음악적으로 더 일관되고 예측 가능한 멜로디를 생성함을 정량적으로 비교합니다.

## 8. 실제 음악 생성 비교

동일한 초기 상태(`seed`)에서 무작위 정책과 학습된 DQN 정책이 생성한 멜로디를 MIDI 파일로 저장하여 비교합니다. 이 과정을 통해 학습된 DQN이 코드 진행에 어울리는 보다 음악적인 멜로디를 생성하는 것을 직접 확인할 수 있습니다.
```
