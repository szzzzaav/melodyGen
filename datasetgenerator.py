import json
import os

import pretty_midi

from play_melody import play_melody

def extract_melody_from_midi(file_path):
    melody_sequence = []
    for dirname, _, filenames in os.walk(file_path):
        for filename in filenames:
            if filename.endswith(".mid"):
                path = os.path.join(dirname, filename)
                print(path)
                midi_data = pretty_midi.PrettyMIDI(path)
                for instrument in midi_data.instruments:
                    if not instrument.is_drum:
                        sum_duration = 0
                        counter = 0
                        for note in instrument.notes:
                            # 获取音名，如 "C5"
                            note_name = pretty_midi.note_number_to_name(note.pitch)
                            # 获取音符持续时间
                            duration = round(note.end - note.start, 2)
                            # 生成形如 "C5-1.0" 的格式
                            sum_duration += duration
                            counter += 1
                        sum_duration /= counter
                        muler = 0.5/sum_duration
                        str = ""
                        counter = 0
                        for note in instrument.notes:
                            # 获取音名，如 "C5"
                            note_name = pretty_midi.note_number_to_name(note.pitch)
                            # 获取音符持续时间
                            duration = round(note.end - note.start, 2)
                            if str == "":
                                str += f"{note_name}-{round(duration*muler,2)}"
                            else:
                                str += f", {note_name}-{round(duration * muler, 2)}"
                            counter += 1
                            if counter >= 50:
                                melody_sequence.append(str)
                                str = ""
                                counter = 0

    with open("dataset2.json",'w') as f:
        json.dump(melody_sequence, f,indent=4)

file_path = "fllDataset"
extract_melody_from_midi(file_path)
# print(melody_sequence)

# play_melody(melody_sequence)
