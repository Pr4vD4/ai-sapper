import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import random
from collections import deque
from config import REWARD_CONFIG, LOGGING_CONFIG, TRAINING_CONFIG  # Добавляем импорт конфигурации наград и LOGGING_CONFIG
from optimizations import calculate_state_vector, calculate_adjacent_value, find_safe_moves_fast
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Dropout
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

class MinesweeperNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256):
        """
        Инициализация нейронной сети
        
        Args:
            input_size: размер входного слоя (ширина * высота * 2)
            hidden_size: размер скрытого слоя
        """
        super().__init__()
        
        # Создаем полносвязную нейронную сеть
        self.network = nn.Sequential(
            # Входной слой
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Скрытые слои
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Выходной слой
            nn.Linear(hidden_size * 2, input_size // 2),
            nn.LayerNorm(input_size // 2),
            nn.ReLU(),
            
            nn.Linear(input_size // 2, input_size // 2)  # Q-значения для каждой клетки
        )
        
        # Инициализация весов
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Убеждаемся, что входные данные имеют нужную размерность
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Добавляем размерность батча
        
        # Добавляем проверку размерности
        if x.size(0) == 1 and self.training:
            # Если в режиме обучения и batch_size=1, дублируем входные данные
            x = x.repeat(2, 1)
            output = self.network(x)
            output = output[0].unsqueeze(0)
        else:
            output = self.network(x)
        
        return output.squeeze() if output.size(0) == 1 else output

    def build_model(self):
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.3),  # Добавляем для предотвращения переобучения
            Dense(self.output_size, activation='sigmoid')
        ])
        return model

class MinesweeperAI:
    def __init__(self, width: int, height: int, num_mines: int):
        """
        Инициализация ИИ для игры в сапёр
        
        Args:
            width: ширина игрового поля
            height: высота игрового поля
            num_mines: количество мин
        """
        self.width = width
        self.height = height
        self.num_mines = num_mines
        
        # Добавляем параметры из конфига
        self.batch_size = TRAINING_CONFIG['BATCH_SIZE']
        self.memory_size = TRAINING_CONFIG['MEMORY_SIZE']
        self.gamma = TRAINING_CONFIG['GAMMA']
        
        # Определяем устройство (GPU или CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device}")
        
        input_size = width * height * 2
        
        # Создаем нейронную сеть и переносим её на GPU
        self.model = MinesweeperNet(input_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=TRAINING_CONFIG['LEARNING_RATE'])
        self.criterion = nn.SmoothL1Loss()
        
        # Добавляем scheduler для динамического изменения learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
        
        # Инициализируем память с указанным размером
        self.memory = deque(maxlen=self.memory_size)
        self.batch_losses = []
        
        # Добавляем целевую сеть
        self.target_model = MinesweeperNet(input_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.update_target_counter = 0
        
        self._state_cache = {}  # Кэш для состояний
        self._move_cache = {}   # Кэш для ходов
        
        # Добавляем scaler для mixed precision training
        self.scaler = GradScaler()
        
        # Добавляем кэш для батчей
        self.states_cache = {}
        self.batch_cache = None
        
    def get_state(self, game) -> np.ndarray:
        """Кэшированная версия получения состояния"""
        board_hash = tuple((cell.is_revealed, cell.adjacent_mines) 
                          for row in game.board 
                          for cell in row)
        
        if board_hash in self._state_cache:
            return self._state_cache[board_hash]
        
        # Создаем numpy массивы для быстрых операций
        revealed_array = np.array([[cell.is_revealed for cell in row] for row in game.board], dtype=np.bool_)
        adjacent_mines_array = np.array([[cell.adjacent_mines for cell in row] for row in game.board], dtype=np.int8)
        
        # Используем оптимизированную функцию
        state = calculate_state_vector(revealed_array, adjacent_mines_array, self.width, self.height)
        
        # Кэшируем результат
        self._state_cache[board_hash] = state
        return state

    def find_safe_moves(self, game) -> List[Tuple[int, int]]:
        """
        Находит все безопасные ходы на текущем поле
        
        Args:
            game: текущее состояние игры
            
        Returns:
            List[Tuple[int, int]]: список координат безопасных ходов
        """
        revealed_array = np.array([[cell.is_revealed for cell in row] for row in game.board], dtype=np.bool_)
        adjacent_mines_array = np.array([[cell.adjacent_mines for cell in row] for row in game.board], dtype=np.int8)
        
        safe_moves = []
        
        # Проверяем все нераскрытые клетки
        for y in range(self.height):
            for x in range(self.width):
                if not revealed_array[y, x]:  # Если клетка не открыта
                    # Проверяем соседние клетки
                    is_safe = False
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                if revealed_array[ny, nx] and adjacent_mines_array[ny, nx] == 0:
                                    is_safe = True
                                    break
                    if is_safe:
                        break
                    
                    if is_safe:
                        safe_moves.append((x, y))
        
        return safe_moves

    def choose_action(self, game, epsilon: float = 0.1) -> Tuple[int, int]:
        """Оптимизированная версия выбора действия"""
        if random.random() < epsilon:  # Исследование
            available_moves = [(x, y) for y in range(self.height) 
                             for x in range(self.width) 
                             if not game.board[y][x].is_revealed]
            return random.choice(available_moves)
        
        # Эксплуатация: оцениваем все возможные ходы сразу
        state = self.get_state(game)
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Создаем маску для доступных ходов
        available_moves = []
        move_mask = torch.zeros(self.width * self.height).to(self.device)
        
        for y in range(self.height):
            for x in range(self.width):
                if not game.board[y][x].is_revealed:
                    idx = y * self.width + x
                    available_moves.append((x, y))
                    move_mask[idx] = 1
        
        with torch.no_grad():
            self.model.eval()  # Переключаем в режим оценки
            q_values = self.model(state_tensor)
            self.model.train()  # Возвращаем в режим обучения
            
            # Маскируем недоступные ходы
            q_values = q_values * move_mask
            best_move_idx = q_values.argmax().item()
            x = best_move_idx % self.width
            y = best_move_idx // self.width
            
        return (x, y)

    def is_safe_move(self, game, x: int, y: int) -> bool:
        """Оптимизированная версия проверки безопасности хода"""
        revealed_array = np.array([[cell.is_revealed for cell in row] for row in game.board], dtype=np.bool_)
        adjacent_mines_array = np.array([[cell.adjacent_mines for cell in row] for row in game.board], dtype=np.int8)
        
        # Используем find_safe_moves_fast вместо find_safe_moves
        safe_moves = find_safe_moves_fast(revealed_array, adjacent_mines_array, self.width, self.height)
        return (x, y) in safe_moves

    def get_adjacent_value(self, game, x: int, y: int) -> float:
        """
        Оптимизированная версия получения значения соседних клеток
        """
        revealed_array = np.array([[cell.is_revealed for cell in row] for row in game.board], dtype=np.bool_)
        adjacent_mines_array = np.array([[cell.adjacent_mines for cell in row] for row in game.board], dtype=np.int8)
        
        return calculate_adjacent_value(revealed_array, adjacent_mines_array, x, y, self.width, self.height)

    def remember(self, state, action, reward, next_state, done):
        """
        Сохраняет опыт в памяти для последующего обучения
        
        Args:
            state: текущее состояние игры (numpy array)
            action: выполненное действие (tuple: x, y)
            reward: полученная награда (float)
            next_state: следующее состояние после действия (numpy array)
            done: флаг завершения игры (bool)
        """
        # Вычисляем приоритет для опыта
        priority = abs(reward)  # Простой способ: используем награду как приоритет
        if done:
            priority *= 2  # Увеличиваем приоритет терминальных состояний
        
        self.memory.append((state, action, reward, next_state, done, priority))

    def train(self, batch_size: int = 32, gamma: float = 0.95):
        """Улучшенная функция обучения"""
        if len(self.memory) < batch_size:
            return
        
        # Выбираем опыт с приоритетом
        experiences = sorted(self.memory, key=lambda x: x[5], reverse=True)[:batch_size]
        # Исправляем распаковку, игнорируя priority
        states, actions, rewards, next_states, dones, _ = zip(*experiences)
        
        # Преобразуем в тензоры
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor([(a[1] * self.width + a[0]) for a in actions]).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1]
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards + (gamma * next_q_values.squeeze() * ~dones)
        
        # Получаем текущие Q-значения
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Вычисляем loss с использованием Huber Loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        # Оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        # Градиентный клиппинг для стабильности
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Обновляем целевую сеть
        if self.update_target_counter % TRAINING_CONFIG['TARGET_UPDATE_FREQ'] == 0:
            self.update_target_network()
        self.update_target_counter += 1

    def calculate_reward(self, game, action: Tuple[int, int], prev_revealed: int) -> float:
        """Улучшенная система наград"""
        x, y = action
        cell = game.board[y][x]
        
        if game.won:
            return 100.0  # Большая награда за победу
        
        if game.game_over:
            return -100.0  # Большой штраф за проигрыш
        
        # Награда за открытие безопасных клеток
        cells_revealed = sum(1 for row in game.board 
                            for c in row if c.is_revealed) - prev_revealed
        
        base_reward = cells_revealed * 5.0  # Базовая награда за открытие клеток
        
        # Дополнительные награды
        if cell.adjacent_mines == 0:  # Открыли пустую клетку
            base_reward += 10.0
        elif cell.adjacent_mines > 0:  # Открыли клетку с числом
            base_reward += 5.0
        
        # Штраф за рискованные ходы
        if not self.is_safe_move(game, x, y):
            base_reward -= 2.0
        
        return base_reward

    def play_episode(self, game, epsilon: float = 0.1, training: bool = True) -> Tuple[float, bool]:
        """
        Проигрывает один эпизод игры
        
        Args:
            game: игровое поле
            epsilon: параметр исследования
            training: флаг режима обучения
        """
        total_reward = 0
        game_over = False
        self.moves = 0
        
        while not game_over:
            self.moves += 1
            current_state = self.get_state(game)
            prev_revealed = sum(1 for row in game.board 
                              for cell in row if cell.is_revealed)
            
            action = self.choose_action(game, epsilon)
            game.reveal(action[0], action[1])
            
            reward = self.calculate_reward(game, action, prev_revealed)
            next_state = self.get_state(game)
            game_over = game.game_over
            
            if training:
                self.remember(current_state, action, reward, next_state, game_over)
                
                if len(self.memory) >= self.batch_size:
                    self.train(batch_size=self.batch_size)
            
            total_reward += reward
        
        return total_reward, game.won

    def update_target_network(self):
        """Обновление весов целевой сети"""
        self.target_model.load_state_dict(self.model.state_dict()) 

    def save_model(self, path: str):
        """Сохранение модели"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
        }, path)

    def load_model(self, path: str):
        """Загрузка модели"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict']) 

    def get_current_lr(self) -> float:
        """Получить текущую скорость обучения"""
        return self.optimizer.param_groups[0]['lr'] 

    def get_curriculum_config(self, episode: int) -> dict:
        """Возвращает конфигурацию игры в зависимости от прогресса обучения"""
        if episode < 500:
            return {'WIDTH': 4, 'HEIGHT': 4, 'NUM_MINES': 2}
        elif episode < 1000:
            return {'WIDTH': 5, 'HEIGHT': 5, 'NUM_MINES': 3}
        else:
            return {'WIDTH': 6, 'HEIGHT': 6, 'NUM_MINES': 5} 

    def pretrain_on_safe_moves(self, num_games=1000):
        """Предварительное обучение на безопасных ходах"""
        for _ in range(num_games):
            game = Minesweeper(self.width, self.height, self.num_mines)
            state = self.get_state(game)
            
            # Первый ход всегда в центр
            x, y = self.width // 2, self.height // 2
            game.reveal(x, y)
            
            while not game.game_over:
                safe_moves = self.find_safe_moves(game)
                if not safe_moves:
                    break
                    
                next_move = random.choice(safe_moves)
                next_state = self.get_state(game)
                reward = 1.0  # Награда за безопасный ход
                
                self.remember(state, next_move, reward, next_state, False)
                if len(self.memory) >= self.batch_size:
                    self.train(self.batch_size)
                    
                state = next_state
                game.reveal(next_move[0], next_move[1]) 