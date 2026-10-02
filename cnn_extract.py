import os
import librosa
import numpy as np
from tqdm import tqdm
import json

def extract_features_cnn(file_path, sr=44100, duration=10,n_mels=128, target_cols=300):
    """
    Extracts Mel-spectrograms for CNN model with a fixed size.
    This version includes a smaller target_cols to speed up training.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)

        y = librosa.util.fix_length(y, size=sr * duration)
        
        # Create Mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)


        mel_spectrogram_fixed = librosa.util.fix_length(mel_spectrogram_db, size=target_cols, axis=1)

 
        return mel_spectrogram_fixed[..., np.newaxis]

    except Exception as e:
        print(f"❌ Error extracting Mel-spectrogram from {file_path}: {e}")
        return None

def process_and_save_data_cnn(base_path, classes, output_dir="cnn_features", output_manifest="cnn_data_manifest.json"):
    """
    Processes audio files, extracts Mel-spectrograms, and saves them.
    This version saves each feature as a separate file and creates a manifest.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    manifest = []
    print("🔄 เริ่มสกัดฟีเจอร์สำหรับ CNN...")

    for idx, cls in enumerate(classes):
        folder = os.path.join(base_path, cls)
        print(f"📁 กำลังประมวลผลคลาส: {cls}")
        for file in tqdm(os.listdir(folder), desc=f"🔍 {cls}", unit="file"):
            if file.endswith(".wav"):
                path = os.path.join(folder, file)
                features = extract_features_cnn(path)
                if features is not None:
                    # Save the feature to a separate .npy file
                    filename = f"{cls}_{os.path.splitext(file)[0]}.npy"
                    filepath = os.path.join(output_dir, filename)
                    np.save(filepath, features)
                    
                    # Add to manifest
                    manifest.append({   
                        "filepath": filepath,
                        "label": idx,
                        "class_name": cls
                    })
    
    with open(output_manifest, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"✅ บันทึกไฟล์ manifest ที่ {output_manifest}")   
    print("✅ สกัดฟีเจอร์เสร็จสิ้น")
    

if __name__ == "__main__":
    BASE_PATH = r"C:\Users\Acer\Downloads\Cough Detection\public_dataset\verify_sound"
    CLASSES = ["covid", "healthy", "symptomatic"]
    process_and_save_data_cnn(BASE_PATH, CLASSES)
