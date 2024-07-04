# 1. Prapemrosesan Data
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Memuat dataset
data = pd.read_csv('music_data.csv')

# Mengonversi kolom kategorikal ke bentuk numerik
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Melihat data yang sudah diproses
print(data.head())


# 2. Definisi Lingkungan RL
import random

class MusicRecommenderEnv:
    def __init__(self, data):
        self.data = data
        self.user_index = 0
        self.state = self.data.iloc[self.user_index]

    def reset(self):
        self.user_index = 0
        self.state = self.data.iloc[self.user_index]
        return self.state

    def step(self, action):
        # Asumsi action adalah indeks dari lagu yang direkomendasikan
        reward = self._get_reward(action)
        self.user_index += 1
        done = self.user_index >= len(self.data)
        if not done:
            self.state = self.data.iloc[self.user_index]
        return self.state, reward, done

    def _get_reward(self, action):
        # Misalnya, reward diukur berdasarkan rating rekomendasi musik (music_recc_rating)
        user = self.data.iloc[self.user_index]
        if user['fav_music_genre'] == action:
            return user['music_recc_rating']
        else:
            return 0


# 3. Pengembangan Agen RL
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.alpha * (target - predict)


# 4. Pelatihan Agen
n_states = len(data)
n_actions = len(data['fav_music_genre'].unique())
agent = QLearningAgent(n_states, n_actions)
env = MusicRecommenderEnv(data)

for episode in range(100):
    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state.name)
        next_state, reward, done = env.step(action)
        agent.learn(state.name, action, reward, next_state.name)
        state = next_state

# Melihat tabel Q yang sudah dilatih
print(agent.q_table)


# 5. Penggunaan Agen untuk Rekomendasi
def recommend_music(user_data):
    state = user_data
    action = agent.choose_action(state.name)
    recommended_genre = label_encoders['fav_music_genre'].inverse_transform([action])
    return recommended_genre

# Contoh rekomendasi untuk pengguna baru
new_user_data = data.iloc[0]  # Sebagai contoh, menggunakan data pengguna pertama
recommended_genre = recommend_music(new_user_data)
print(f"Rekomendasi genre musik untuk pengguna baru: {recommended_genre}")