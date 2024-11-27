from play_melody import play_melody
import librosa
import numpy as np


# 音高转换表
def hz_to_note(hz):
    """将频率转换为音符"""
    if hz == 0 or np.isnan(hz):
        return None  # 无音高（静音）
    midi_num = int(round(12 * np.log2(hz / 440.0) + 69))
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = midi_num // 12 - 1
    note = note_names[midi_num % 12] + str(octave)
    return note


def smooth_notes(pitches, magnitudes, hop_length, sr, smooth_window=5):
    """对提取的音符进行平滑处理"""
    smoothed_notes = []
    for frame, pitch in enumerate(pitches.T):
        # 提取当前帧的最强频率分量
        hz = pitch[np.argmax(magnitudes[:, frame])]
        note = hz_to_note(hz)
        smoothed_notes.append(note if note else "REST")

    # 应用滑动窗口进行平滑
    for i in range(len(smoothed_notes)):
        window_start = max(0, i - smooth_window // 2)
        window_end = min(len(smoothed_notes), i + smooth_window // 2 + 1)
        window_notes = smoothed_notes[window_start:window_end]
        # 取窗口中最常出现的音符作为平滑结果
        smoothed_notes[i] = max(set(window_notes), key=window_notes.count)

    return smoothed_notes


def extract_melody(file_path, hop_length=2048, smooth_window=5):
    """从音频文件提取旋律"""
    y, sr = librosa.load(file_path, sr=None)  # 加载音频文件
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)  # 提取音高

    # 平滑处理音符
    smoothed_notes = smooth_notes(pitches, magnitudes, hop_length, sr, smooth_window)
    melody = []
    current_note = None
    note_start_time = 0
    frame_duration = hop_length / sr  # 每帧的实际时长

    for frame, note in enumerate(smoothed_notes):
        if note != current_note:
            if current_note and current_note != "REST":
                duration = (frame - note_start_time) * frame_duration
                melody.append(f"{current_note}-{round(duration, 2)}")
            current_note = note
            note_start_time = frame

    # 处理最后一个音符
    if current_note and current_note != "REST":
        duration = (len(smoothed_notes) - note_start_time) * frame_duration
        melody.append(f"{current_note}-{round(duration, 2)}")

    return melody


# 测试提取旋律
file_path = "test.mp3"  # 替换为你的 MP3 文件路径
melody = extract_melody(file_path)
print(melody)

play_melody(melody)
