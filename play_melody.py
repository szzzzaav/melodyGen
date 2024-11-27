from music21 import stream, note, midi

def play_melody(melody):
    """使用 music21 播放旋律"""
    melody_stream = stream.Stream()
    print("finish stream")

    for item in melody:
        try:
            if item.strip().split('-')[0] == "rest":  # 检查是否为休止符
                rest = note.Rest()  # 创建休止符
                rest.quarterLength = float(item.strip().split('-')[1])  # 设置时长
                melody_stream.append(rest)  # 添加到旋律流中
            else:
                # 确保音符和时长能正确解析
                note_name, duration = item.split("-")
                n = note.Note(note_name.strip())  # 去掉多余空格
                n.quarterLength = float(duration.strip())  # 设置时长
                melody_stream.append(n)  # 添加到旋律流中
        except Exception as e:
            print(f"跳过无效条目: {item}, 错误: {e}")

    # 播放旋律
    midi.realtime.StreamPlayer(melody_stream).play()

# 测试播放
# play_melody("C4-1.0 rest-0.5 D4-0.5 E4-1.0".split(" "))
