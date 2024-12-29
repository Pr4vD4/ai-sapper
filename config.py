"""
Конфигурация для обучения и тестирования ИИ в игре Сапёр
"""

# Параметры игрового поля
GAME_CONFIG = {
    'WIDTH': 4,
    'HEIGHT': 4,
    'NUM_MINES': 2
}

# Параметры обучения
TRAINING_CONFIG = {
    'NUM_EPISODES': 8000,
    'BATCH_SIZE': 64,
    'MEMORY_SIZE': 50000,
    'GAMMA': 0.99,
    'INITIAL_EPSILON': 1.0,
    'FINAL_EPSILON': 0.2,
    'EPSILON_DECAY': 0.999,
    'LEARNING_RATE': 0.001,
    'TARGET_UPDATE_FREQ': 10,
    'LOAD_EXISTING_MODEL': False,
    'EPSILON_MIN_EPISODE': 2000,
    'PRIORITIZED_REPLAY': True,
    'DOUBLE_DQN': True,
    'CURRICULUM_LEARNING': False
}

# Параметры нейронной сети
NETWORK_CONFIG = {
    'HIDDEN_SIZE': 512,
    'NUM_LAYERS': 4,
    'DROPOUT': 0.2,
    'LEARNING_RATE': 0.001,
    'TARGET_UPDATE_FREQ': 5
}

# Параметры наград
REWARD_CONFIG = {
    'WIN': 100.0,
    'LOSS': -20.0,
    'SAFE_MOVE': 2.0,
    'HINT_BONUS': 3.0,
    'PROGRESS_BONUS': 5.0,
    'EARLY_GAME_BONUS': 1.0,
    'PATTERN_BONUS': 2.0,
    'RISKY_MOVE_PENALTY': -1.0,
    'CONSECUTIVE_SAFE_BONUS': 1.0,
    'EXPLORATION_BONUS': 0.5,
}

# Параметры логирования и сохранения
LOGGING_CONFIG = {
    'LOG_DIR': 'logs',
    'SAVE_DIR': 'models',  # Директория для сохранения моделей
    'LOG_INTERVAL': 500,   # Интервал для логирования статистики
    'SAVE_CHECKPOINTS': False,  # Сохранять ли промежуточные модели
    'CHECKPOINT_INTERVAL': 200,  # Каждые сколько эпизодов сохранять модель
    'SHOW_WIN_BOARD': True,     # Показывать ли состояние поля при победе
}

CURRICULUM_CONFIG = {
    'ENABLED': False,
    'STAGES': [
        {'EPISODES': 100, 'WIDTH': 4, 'HEIGHT': 4, 'MINES': 2},
        {'EPISODES': 200, 'WIDTH': 5, 'HEIGHT': 5, 'MINES': 3},
        {'EPISODES': 500, 'WIDTH': 6, 'HEIGHT': 6, 'MINES': 5}
    ]
}

# Добавляем новые параметры
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE_DECAY = 0.95
VALIDATION_SPLIT = 0.2 

# Добавляем параметры для оптимизации производительности
PERFORMANCE_CONFIG = {
    'USE_MIXED_PRECISION': True,
    'CACHE_SIZE': 10000,
    'NUM_WORKERS': 4,
    'PREFETCH_FACTOR': 2,
    'PIN_MEMORY': True,
    'ASYNC_LOADING': True,
    'OPTIMIZE_MEMORY': True,
    
    # Параметры для батчей
    'ADAPTIVE_BATCH_SIZE': True,
    'MIN_BATCH_SIZE': 32,
    'MAX_BATCH_SIZE': 256,
    'BATCH_GROWTH_RATE': 1.1
} 