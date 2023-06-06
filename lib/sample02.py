import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Загрузка и предобработка данных
text = open('text_data.txt', 'r').read()
chars = list(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
data_size, vocab_size = len(text), len(chars)

# Гиперпараметры модели
hidden_size = 100
seq_length = 25
learning_rate = 0.01
num_epochs = 100

# Создание тренировочных пар (входной последовательность, целевая последовательность)
input_seqs = []
target_seqs = []
for i in range(0, data_size - seq_length, seq_length):
    input_seqs.append([char_to_idx[ch] for ch in text[i:i+seq_length]])
    target_seqs.append([char_to_idx[ch] for ch in text[i+1:i+seq_length+1]])

# Конвертация входных и целевых последовательностей в тензоры PyTorch
input_seqs = torch.tensor(input_seqs, dtype=torch.long)
target_seqs = torch.tensor(target_seqs, dtype=torch.long)

# Определение модели RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out.view(-1, self.hidden_size))
        return out, hidden

# Инициализация модели и оптимизатора
model = RNN(vocab_size, hidden_size, vocab_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Обучение модели
for epoch in range(num_epochs):
    hidden = None
    loss = 0
    for i in range(0, input_seqs.size(0) - seq_length, seq_length):
        input_batch = input_seqs[i:i+seq_length]
        target_batch = target_seqs[i:i+seq_length]
        
        optimizer.zero_grad()
        
        if hidden is not None:
            if isinstance(hidden, tuple):
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:
                hidden = hidden.detach()
        
        outputs, hidden = model(input_batch, hidden)
        loss = criterion(outputs, target_batch.view(-1))
        
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

print('Обучение закончено!')

# Генерация текста с помощью обученной модели
start_seq = 'The '
generated_text = start_seq
with torch.no_grad():
    hidden = None
    input_tensor = torch.tensor([char_to_idx[ch] for ch in start_seq], dtype=torch.long).unsqueeze(0)
    for _ in range(100):
        outputs, hidden = model(input_tensor, hidden)
        _, top_idx = torch.max(outputs, dim=1)
        generated_char = idx_to_char[top_idx.item()]
        generated_text += generated_char
        input_tensor = torch.tensor([[top_idx.item()]], dtype=torch.long)
        
print('Сгенерированный текст:')
print(generated_text)
