import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import random
from collections import deque
from config import REWARD_CONFIG, LOGGING_CONFIG, TRAINING_CONFIG  # Добавляем импорт конфигурации наград и LOGGING_CONFIG
from optimizations import calculate_state_vector, calculate_adjacent_value, find_safe_moves_fast

class MinesweeperNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 512):
        """
        Инициализация нейронной сети
        
        Args:
            input_size: размер входного слоя (ширина * высота * 2)
            hidden_size: размер скрытого слоя
        """
        super().__init__()
        
        # Создаем полносвязную нейронную сеть
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # Убеждаемся, что входные данные имеют нужную размерность
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Добавляем размерность батча
        return self.network(x)

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
        revealed_array = np.array([[cell.is_revealed for cell in row] for row in game.board], dtype=np.bool_)
        adjacent_mines_array = np.array([[cell.adjacent_mines for cell in row] for row in game.board], dtype=np.int8)
        
        # Первый ход в центр
        if not revealed_array.any():
            return (self.width // 2, self.height // 2)
        
        # Поиск безопасных ходов
        safe_moves = find_safe_moves_fast(revealed_array, adjacent_mines_array, self.width, self.height)
        if safe_moves and random.random() > epsilon * 0.5:
            return random.choice(safe_moves)
        
        if random.random() < epsilon:
            available_moves = [(x, y) for y in range(self.height) for x in range(self.width) 
                              if not revealed_array[y, x]]
            return random.choice(available_moves)
        
        # Оцениваем все ходы одним батчем
        state = self.get_state(game)
        state_tensor = torch.from_numpy(state).to(self.device)
        
        available_moves = []
        states_batch = []
        
        for y in range(self.height):
            for x in range(self.width):
                if not revealed_array[y, x]:
                    available_moves.append((x, y))
                    temp_state = state_tensor.clone()
                    idx = (y * self.width + x) * 2
                    temp_state[idx] = 1.0
                    temp_state[idx + 1] = calculate_adjacent_value(revealed_array, adjacent_mines_array, x, y, self.width, self.height)
                    states_batch.append(temp_state)
        
        if not available_moves:
            raise ValueError("Нет доступных ходов")
        
        # Оцениваем все состояния одним батчем
        states_batch = torch.stack(states_batch)
        with torch.no_grad():
            values = self.model(states_batch).squeeze()
        
        return available_moves[values.argmax().item()]

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
        """Обучает нейронную сеть на основе сохраненного опыта"""
        if len(self.memory) < batch_size:
            return
        
        # Используем numpy для быстрой обработки батча
        batch = random.sample(self.memory, batch_size)
        states = np.vstack([exp[0] for exp in batch])
        next_states = np.vstack([exp[3] for exp in batch])
        
        # Переводим в тензоры один раз
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        # Получаем предсказания одним батчем
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
        
        # Вычисляем целевые значения
        targets = torch.zeros(batch_size, device=self.device)
        for i, (_, _, reward, _, done, _) in enumerate(batch):  # Добавляем _ для priority
            targets[i] = reward if done else reward + gamma * next_q_values[i]
        
        # Обучаем модель
        self.optimizer.zero_grad()
        current_q_values = self.model(states)
        loss = self.criterion(current_q_values.squeeze(), targets)
        loss.backward()
        self.optimizer.step()
        
        self.batch_losses.append(loss.item())
        
        if hasattr(self, 'logger'):
            self.logger.debug(f"Batch loss: {loss.item():.6f}")
            
        # Обновляем целевую сеть каждые 10 батчей
        self.update_target_counter += 1
        if self.update_target_counter % 10 == 0:
            self.update_target_network()

    def calculate_reward(self, game, action: Tuple[int, int], prev_revealed: int) -> float:
        """Оптимизированная версия вычисления наград"""
        x, y = action
        cell = game.board[y][x]
        
        if game.won:
            return REWARD_CONFIG['WIN']
        if game.game_over:
            return REWARD_CONFIG['LOSS']
        
        # Создаем массив для подсчета открытых клеток
        revealed_array = np.array([[cell.is_revealed for cell in row] for row in game.board], dtype=np.bool_)
        current_revealed = np.sum(revealed_array)
        cells_revealed = current_revealed - prev_revealed
        
        reward = REWARD_CONFIG['SAFE_MOVE'] * cells_revealed
        if cell.adjacent_mines == 0:
            reward += REWARD_CONFIG['PROGRESS_BONUS']
        elif cell.adjacent_mines > 0:
            reward += REWARD_CONFIG['HINT_BONUS']
        
        return reward

    def play_episode(self, game, epsilon: float = 0.1) -> Tuple[float, bool]:
        """
        Проигрывает один эпизод игры
        """
        total_reward = 0
        game_over = False
        self.moves = 0  # Добавляем счетчик ходов
        
        while not game_over:
            self.moves += 1
            current_state = self.get_state(game)
            prev_revealed = sum(1 for row in game.board 
                              for cell in row if cell.is_revealed)
            
            # Проверяем условие победы до хода
            safe_unopened = sum(1 for row in game.board 
                              for cell in row 
                              if not cell.is_mine and not cell.is_revealed)
            
            action = self.choose_action(game, epsilon)
            game.reveal(action[0], action[1])
            
            # Проверяем условие победы после хода
            reward = self.calculate_reward(game, action, prev_revealed)
            next_state = self.get_state(game)
            game_over = game.game_over
            
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