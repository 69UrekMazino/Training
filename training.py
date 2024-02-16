import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
import pandas as pd

# Define the dataset class to load video frames and labels
class VideoDataset(Dataset):
    def __init__(self, video_filenames, video_directory, label_directory):
        self.video_filenames = video_filenames
        self.video_directory = video_directory
        self.label_directory = label_directory
        self.frames_and_labels = []

        # Load frames and labels
        for video_file in self.video_filenames:
            video_path = os.path.join(video_directory, video_file)
            label_path = os.path.join(label_directory, os.path.splitext(video_file)[0] + '.csv')
            labels_df = pd.read_csv(label_path, header=None)
            cap = cv2.VideoCapture(video_path)
            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_index < len(labels_df):
                    label = float(labels_df.iloc[frame_index, 0])
                else:
                    label = None
                self.frames_and_labels.append((frame, label))
                frame_index += 1
            cap.release()

        print(f"Length of frames and labels: {len(self.frames_and_labels)}")

    def __len__(self):
        return len(self.frames_and_labels)

    def __getitem__(self, idx):
        frame, label = self.frames_and_labels[idx]
        return frame, label

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Process each frame to extract face landmarks
def extract_face_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = []
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks
    else:
        return None

train_video_directory = 'C:/Users/Acer/Desktop/MD/Dataset/Training'
train_label_directory = 'C:/Users/Acer/Desktop/MD/Dataset/TrainingLabels'

train_video_filenames = os.listdir(train_video_directory)

train_dataset = VideoDataset(train_video_filenames, train_video_directory, train_label_directory)

batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define LSTM model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

input_size = 468  # Number of face landmarks (3 coordinates each)
hidden_size = 64
num_layers = 2
learning_rate = 0.001
num_epochs = 10

# Initialize the LSTM model
model = LSTMModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (videos, labels) in enumerate(train_loader):
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos.float())

        # Calculate loss using Mean Squared Error (MSE)
        loss = criterion(outputs.squeeze(), labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'C:/Users/Acer/Desktop/MD/lstm_model.pth')
